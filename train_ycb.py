#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3, 4, 5, 6, 7'
import warnings
warnings.filterwarnings("ignore")
import random
import time
from copy import deepcopy
import shutil, time
import numpy as np
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter

# from datasets.tless.tless_dataset import PoseDataset as pose_dataset
#from datasets.dttd_iphone.dataset import DTTDDataset as dttd_dataset
from datasets.ycb.ycb_dataset import PoseDataset as ycb_dataset

from models import ES6D as pose_net

# from lib.tless_evaluator import TLESSADDval
# from lib.tless_gadd_evaluator import TLESSGADDval

from lib.utils import setup_logger
from lib.utils import warnup_lr, cal_mean_std
from lib.utils import post_processing_ycb_quaternion as post_processing
from lib.utils import save_pred_and_gt_json

st_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', type=int, default=0, help='gpu number')
parser.add_argument('--experiment', type=str, default= "train",  help='brief description about experiment setting: train, test')
parser.add_argument('--loss_type', type=str, default= "ADD",  help='trianing loss: GADD, ADD')
parser.add_argument('--dataset', type=str, default='dttd', help='ycb, tless')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--workers', type=int, default=8, help='number of data loading workers')
parser.add_argument('--lr', default=2e-7, help='learning rate, note that the learning rate at tless dataset is much larger than the ycb-video dataset')
parser.add_argument('--lr_rate', default=0.1, help='learning rate decay rate')
parser.add_argument('--warnup_iters', default=100, help='learning rate decay rate')
parser.add_argument('--decay_epoch', default=60, help='learning rate decay rate')
parser.add_argument('--noise_trans', default=0.03, help='range of the random noise of translation added to the training data')
parser.add_argument('--augmentation', type=bool, default= False, help='train tless with data augmentation or not')
parser.add_argument('--nepoch', type=int, default=120, help='max number of epochs to train') #
parser.add_argument('--start_epoch', type=int, default=0, help='which epoch to start') #

opt = parser.parse_args()


def main():

    # pre-setup
    global opt

    torch.backends.cudnn.enabled = True
    opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '1235'

    opt.gpu_number = torch.cuda.device_count()

    if opt.dataset == 'ycb':
        opt.num_objects = 21  # number of object classes in the dataset
        opt.num_points = 1024
        opt.dataset_root = './datasets/ycb'

        opt.outf = 'experiments/ycb/{}/{}/model/'.format(opt.loss_type, opt.experiment)  # folder to save trained model
        opt.log_dir = 'experiments/ycb/{}/{}/log/'.format(opt.loss_type, opt.experiment)  # folder to save logs ########

        if os.path.isdir(opt.outf) == False:
            os.makedirs(opt.outf)

        if os.path.isdir(opt.log_dir) == False:
            os.makedirs(opt.log_dir)

    mp.spawn(per_processor, nprocs=opt.gpu_number, args=(opt,))



def predict(data, estimator, lossor, opt, mode='train'):


    cls_ids = data['class_id'].to(opt.gpu)
    rgb = data['rgb'].to(opt.gpu)
    depth = data['xyz'].to(opt.gpu)
    mask = data['mask'].to(opt.gpu)
    gt_r = data['target_r'].to(opt.gpu)
    gt_t = data['target_t'].to(opt.gpu)

    model_xyz = data['model_xyz'].cpu().numpy()
    preds = estimator(rgb, depth, cls_ids)

    if mode == 'train':
        loss, loss_dict = lossor(preds, mask, gt_r, gt_t,  cls_ids, model_xyz)
        return loss, loss_dict
    elif mode == 'test':
        return preds["pred_r"], preds["pred_t"]
    else:
        raise ValueError("Invalid mode")


def per_processor(gpu, opt):
    opt.gpu = gpu
    tensorboard_writer = 0
    if gpu == 0:
        tensorboard_writer = SummaryWriter(opt.log_dir)

    # init processor
    
    torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:23456', rank=gpu, world_size=opt.gpu_number)
    print("init gps:{}".format(gpu))
    torch.cuda.set_device(gpu)


    # init DDP model
    estimator = pose_net.ES6D(num_class=opt.num_objects).to(gpu)
    estimator = torch.nn.parallel.DistributedDataParallel(estimator, device_ids=[gpu], output_device=gpu, find_unused_parameters=True)

    # init optimizer
    optimizer = optim.Adam(estimator.parameters(), lr=opt.lr * opt.gpu_number)

    # init DDP dataloader
    # dataset = pose_dataset('train', opt.num_points, opt.dataset_root, True, opt.noise_trans)
    # dataset = dttd_dataset('./datasets/dttd_iphone/DTTD_IPhone_Dataset/root', mode='train', config_path='./datasets/dttd_iphone/dataset_config')
    dataset = ycb_dataset(mode ='train', num_pt = opt.num_points, root = opt.dataset_root, add_noise=False, noise_trans=opt.noise_trans)


    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=False,
                                             num_workers=opt.workers, pin_memory=True, sampler=sampler)

    # if gpu == 0:

    #     # test_set = pose_dataset('test', opt.num_points, opt.dataset_root, False, opt.noise_trans)
    #     test_set = dttd_dataset('./datasets/dttd_iphone/DTTD_IPhone_Dataset/root', mode='test', config_path='./datasets/dttd_iphone/dataset_config')
    #     test_loader = torch.utils.data.DataLoader(test_set, batch_size=opt.batch_size, shuffle=False,
    #                                              num_workers=opt.workers*2, pin_memory=True)


    opt.sym_list = dataset.get_sym_list()
    opt.num_points_mesh = dataset.get_num_points_mesh()


    opt.obj_radius = dataset.obj_radius
    opt.raw_prim_groups = dataset.raw_prim_groups


    # init loss model
    train_loss = pose_net.get_loss(dataset = dataset, loss_type= opt.loss_type, train = True).to(gpu)
    # test_loss = pose_net.get_loss(dataset=dataset, loss_type=opt.loss_type, train = False).to(gpu)


    # epoch loop
    tensorboard_loss_list = []
    tensorboard_test_list = []


    for epoch in range(opt.start_epoch, opt.nepoch + 1):


        sampler.set_epoch(epoch)
        opt.cur_epoch = epoch

        # # train for one epoch
        print('>>>>>>>>>>>train>>>>>>>>>>>')
        train(trainloader, estimator, train_loss, optimizer, epoch, tensorboard_writer, tensorboard_loss_list, opt)
        torch.cuda.empty_cache()

        # save checkpoint
        if gpu == 0 and epoch % 5 == 0:
            print('>>>>>>>>>>>save checkpoint>>>>>>>>>>')
            torch.save({
                'epoch': epoch + 1,
                'state_dict': estimator.state_dict()},
                '{}/checkpoint_{:04d}.pth.tar'.format(opt.outf, epoch))


        # # test for one epoch
        # if gpu == 0 and epoch % 5 == 0:
        #     print('>>>>>>>>>>>test>>>>>>>>>>>')
        #     test(test_loader, estimator, test_loss, epoch, tensorboard_writer, tensorboard_test_list, opt)
        #     torch.cuda.empty_cache()




# def test(test_loader, estimator, lossor, epoch, tensorboard_writer, tensorboard_test_list, opt):

#     if opt.gpu == 0:
#         test_loss_list = []
#         logger = setup_logger('epoch%d' % epoch, os.path.join(opt.log_dir, 'test_%d_log.txt' % epoch))

#         logger.info('total test number : {}'.format(len(test_loader)))

#         # init evaluator
#         tless_add_evaluator = TLESSADDval()
#         # tless_gadd_evaluator = TLESSGADDval()

#     estimator.eval()

#     raw_prim_groups_set = opt.raw_prim_groups

#     with torch.no_grad():

#         i = 0
#         total_rt_list = []
#         total_gt_list = []
#         total_cls_list = []
#         total_instance_list = []
#         total_RT_list = []

#         for data in test_loader:

#             i += 1

#             _, test_loss_dict, rt_list, gt_rt_list, gt_cls_list, model_list, instance_id_list, instance_eval_rt_list  = predict(data, estimator, lossor, opt, mode='test')

#             total_rt_list += rt_list
#             total_gt_list += gt_rt_list
#             total_cls_list += gt_cls_list
#             total_instance_list += instance_id_list
#             total_RT_list += instance_eval_rt_list

#             # eval
#             tless_add_evaluator.eval_pose_parallel(rt_list, gt_cls_list, gt_rt_list, gt_cls_list, model_list)

#             model_list = []
#             for gt_cls in gt_cls_list:
#                 model_list.append(raw_prim_groups_set[gt_cls[0]-1])

#             # tless_gadd_evaluator.eval_pose_parallel(rt_list, gt_cls_list, gt_rt_list, gt_cls_list, model_list)


#             # log and draw loss
#             if opt.gpu == 0:
#                 # log
#                 test_loss_list.append(test_loss_dict)
#                 log_function(test_loss_list, logger, epoch, i, opt.lr)

#         save_pred_and_gt_json(total_RT_list, total_instance_list, total_gt_list, total_cls_list, opt.log_dir)

#         # draw loss
#         if opt.gpu == 0:

#             # evaluation result
#             add_cur_eval_info_dict = tless_add_evaluator.cal_auc()
#             # gadd_cur_eval_info_dict = tless_gadd_evaluator.cal_auc()

#             # draw
#             l = deepcopy(test_loss_list[0])

#             for ld in test_loss_list[1:]:
#                 for key in ld:
#                     l[key] += ld[key]
#             for key in l:
#                 l[key] = l[key] / len(test_loss_list)

#             l['add_auc'] = add_cur_eval_info_dict['auc']
#             # l['gadd_auc'] = gadd_cur_eval_info_dict['auc']

#             tensorboard_test_list.append(l)
#             draw_loss_list('test', tensorboard_test_list, tensorboard_writer)

#             # output test result
#             log_tmp = 'TEST ENDING: '

#             for key in l:

#                 log_tmp = log_tmp + ' {}:{:.4f}'.format(key, l[key])

#             logger.info(log_tmp)


def train(train_loader, estimator, lossor, optimizer, epoch, tensorboard_writer, tensorboard_loss_list, opt):

    if opt.gpu == 0:
        train_loss_list = []
        logger = setup_logger('epoch%d' % epoch, os.path.join(opt.log_dir, 'epoch_%d_log.txt' % epoch))

        if opt.gpu == 0:

            for key, value in sorted(vars(opt).items()):
                logger.info(str(key) + ': ' + str(value))

            # record
            logger.info('total train number : {}'.format(len(train_loader)) )


    estimator.train()
    optimizer.zero_grad()

    i = 0
    for data in train_loader:
        try:
            i += 1
            # update learning rate
            iter_th = epoch * len(train_loader) + i

            cur_lr = adjust_learning_rate(optimizer, epoch, iter_th, opt)

            loss, loss_dict = predict(data, estimator, lossor, opt, mode='train')


            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # log and draw loss
            if opt.gpu == 0:

                train_loss_list.append(loss_dict)
                log_function(train_loss_list, logger, epoch, i, cur_lr)

                if len(train_loss_list) % 50 == 0:
                    l_dict = deepcopy(train_loss_list[-50])
                    for ld in train_loss_list[-49:]:
                        for key in ld:
                            l_dict[key] += ld[key]
                    for key in l_dict:
                        l_dict[key] = l_dict[key] / 50.0

                    tensorboard_loss_list.append(l_dict)
                    draw_loss_list('train', tensorboard_loss_list, tensorboard_writer)
        except:
            print(data['file_index'])




def adjust_learning_rate(optimizer, epoch, iter, opt):

    """Decay the learning rate based on schedule"""
    lr = opt.lr * opt.gpu_number

    lr *= opt.lr_rate if epoch >= opt.decay_epoch else 1.

    if (iter <= opt.warnup_iters):
        lr = warnup_lr(iter, opt.warnup_iters, lr / 10, lr)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

def log_function(loss_list, logger, epoch, batch, lr):
    l = loss_list[-1]
    tmp = 'time{} E{} B{} lr:{:.9f}'.format(
        time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), epoch, batch, lr)
    for key in l:
        tmp = tmp + ' {}:{:.4f}'.format(key, l[key])
    logger.info(tmp)

def draw_loss_list(phase, loss_list, tensorboard_writer):

    loss = loss_list[-1]

    for key in loss:

        tensorboard_writer.add_scalar(phase+'/'+key, loss[key], len(loss_list))


if __name__ == '__main__':

    main()

    # envs mypose

