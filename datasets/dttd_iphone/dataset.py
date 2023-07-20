import json
import os
import random
import time
from cgi import test

import cv2
import numpy as np
import numpy.ma as ma
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

Borderlist = [-1] + list(range(40, 1960, 40))

class DTTDDataset(data.Dataset):
    def __init__(self, root, mode, refine=False, add_noise=True, config_path="/ssd/zixun/Documents/DigitalTwin-6DPose_cleaned/estimation/dataset/dttd_iphone/dataset_config"):
        self.Trainlist = os.path.join(config_path, 'train_data_list.txt')
        self.Testlist = os.path.join(config_path, 'test_data_list.txt')
        self.Classlist = os.path.join(config_path, 'objectids.csv')
        self.mode = mode
        self.root = root
        self.refine = refine
        self.add_noise = add_noise
        self._init_config()
        self._init_data_info()
        self.model_points = self.load_points(self.root, self.classes) # {1:[[x1, y1, z1], [x2, y2, z2], ...], 2:[...], 3:[...], ...}

    def __len__(self):
        return len(self.all_data_dirs)

    def __getitem__(self, index):
        # load raw data
        img = Image.open(os.path.join(self.root, self.prefix, f"{self.all_data_dirs[index]}_color.jpg")) 
        depth = np.array(Image.open(os.path.join(self.root, self.prefix, f"{self.all_data_dirs[index]}_depth.png")), dtype=np.uint16) 
        label = np.array(Image.open(os.path.join(self.root, self.prefix, f"{self.all_data_dirs[index]}_label.png"))) 
        with open(os.path.join(self.root, self.prefix, f"{self.all_data_dirs[index]}_meta.json"), "r") as f:
            # dict_keys(['objects', 'object_poses', 'intrinsic', 'distortion'])
            meta = json.load(f)
        
        # set camera focus and center
        cam = np.array(meta['intrinsic'])
        cam_cx, cam_cy, cam_fx, cam_fy =  cam[0][2], cam[1][2], cam[0][0], cam[1][1]
        
        # color jittor (noise) for img
        if self.add_noise:
            img = self.trancolor(img)
        img = np.array(img) # shape: (1440, 1920, 3)

        # Get id of objects that appear in the image
        objs = np.array(meta['objects']).flatten().astype(np.int32)
        
        if len(objs) == 0:
            print(self.all_data_dirs[index])
        # randomly choose one object and get its mask
        obj_indices = list(range(len(objs)))
        random.shuffle(obj_indices)
        for obj_idx in obj_indices:
            # consider both label (where = objs[obj_idx]) and valid depth (where > 0)
            mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
            mask_label = ma.getmaskarray(ma.masked_equal(label, objs[obj_idx]))
            mask = mask_label * mask_depth if self.use_labelmask else mask_depth
            if self.debug_mode: 
                cv2.imwrite(f"test_{objs[obj_idx]}.png", mask.astype(np.int32)*255)
            if len((mask_label * mask_depth).nonzero()[0]) > self.minimum_px_num:
                break

        # get ground truth rotation and translation
        R_gt = np.array(meta['object_poses'][str(objs[obj_idx])])[0:3, 0:3] # (3, 3)
        T_gt = np.array(meta['object_poses'][str(objs[obj_idx])])[0:3, 3:4].flatten().reshape(1, 3) # (1, 3)

        # get sample points (3D) on model
        model_sample_list = list(range(len(self.model_points[objs[obj_idx]])))
        if self.refine:
            model_sample_list = sorted(random.sample(model_sample_list, self.pt_num_mesh_large))
        else:
            model_sample_list = sorted(random.sample(model_sample_list, self.pt_num_mesh_small))
        sampled_model_pt = np.array(self.model_points[objs[obj_idx]][model_sample_list, :]) # (pt_num_mesh*, 3) 

        # get model points in the world coordinate
        sampled_model_pt_world = np.add(np.dot(sampled_model_pt, R_gt.T), T_gt) # (pt_num_mesh*, 3)
        
        # get object's bounding box
        img_h, img_w = label.shape
        rmin, rmax, cmin, cmax = get_discrete_width_bbox(mask_label, Borderlist, img_w, img_h)
        
        # set sample points
        # set sample points (2D) on depth/point cloud
        sample2D = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0] # non-zero positions on flattened mask, 1-D array
        if len(sample2D) >= self.sample_2d_pt_num:
            sample2D = np.array(sorted(np.random.choice(sample2D, self.sample_2d_pt_num))) # randomly choose sample_2d_pt_num points (idx)
        elif len(sample2D) == 0:
            sample2D = np.pad(sample2D, (0, self.sample_2d_pt_num - len(sample2D)), 'constant')
        else:
            sample2D = np.pad(sample2D, (0, self.sample_2d_pt_num - len(sample2D)), 'wrap')

        # crop image, depth and xy map with bbox, take the sample points
        img_crop = img[:, :, :3][rmin:rmax, cmin:cmax, :]
        depth_crop = depth[rmin:rmax, cmin:cmax].astype(np.float32) # (h, w, )
        mask_crop = mask[rmin:rmax, cmin:cmax].astype(np.float32) # (h, w, )
        xmap_crop = self.xmap[rmin:rmax, cmin:cmax].astype(np.float32) # (h, w, ) store y for sample points
        ymap_crop = self.ymap[rmin:rmax, cmin:cmax].astype(np.float32) # (h, w, ) store x for sample points
        
        # get point cloud [[px, py, pz], ...]
        cam_scale = 1000 # uint16 * 1000
        pz = depth_crop / cam_scale
        px = (ymap_crop - cam_cx) * pz / cam_fx
        py = (xmap_crop - cam_cy) * pz / cam_fy
        px, py, pz = np.expand_dims(px, axis=2), np.expand_dims(py, axis=2), np.expand_dims(pz, axis=2)
        xyz_map = np.concatenate((px, py, pz), axis=2) # (h, w, 3) store XYZ point cloud value for sample points
        
        # calculate mean xyz
        mask_x = xyz_map[:, :, 0].flatten()[sample2D][:, np.newaxis]
        mask_y = xyz_map[:, :, 1].flatten()[sample2D][:, np.newaxis]
        mask_z = xyz_map[:, :, 2].flatten()[sample2D][:, np.newaxis]
        mask_xyz = np.concatenate((mask_x, mask_y, mask_z), axis=1)
        mean_xyz = mask_xyz.mean(axis=0).reshape((3))

        xyz_map = (xyz_map - mean_xyz) * mask_crop[:, :, np.newaxis]
        # xyz_map = (xyz_map) * mask_crop[:, :, np.newaxis]

        # add noise for pointcloud and model points
        if self.add_noise:
            add_noise_t = np.array([random.uniform(-self.noise_trans, self.noise_trans) for i in range(3)])
            xyz_map = np.add(xyz_map, add_noise_t)
            sampled_model_pt_world = np.add(sampled_model_pt_world, add_noise_t)
        
        xyz_map = np.transpose(xyz_map, (2,0,1))
        img_crop = np.transpose(img_crop, (2,0,1))

        return {
               "xyz": torch.from_numpy(xyz_map.astype(np.float32)), \
               "mask": torch.from_numpy(mask_crop.astype(np.float32)).unsqueeze(dim=0), \
               "rgb": self.norm(torch.from_numpy(img_crop.astype(np.float32))), \
               "sampled_model_pt_camera": torch.from_numpy(sampled_model_pt_world.astype(np.float32)), \
               "model_xyz": torch.from_numpy(sampled_model_pt.astype(np.float32)), \
               "class_id": torch.LongTensor([int(objs[obj_idx]) - 1]), \
               "target_r": torch.from_numpy(R_gt.astype(np.float32)), \
               "target_t": torch.from_numpy(T_gt.reshape(3, -1).astype(np.float32)), \
               "file_index": self.all_data_dirs[index], \
               "mean_xyz": mean_xyz
               } 

    def _init_config(self):
        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        self.xmap = np.array([[j for i in range(1920)] for j in range(1440)])
        self.ymap = np.array([[i for i in range(1920)] for j in range(1440)])

        self.noise_trans = 0.02
        self.minimum_px_num = 50
        self.symmetry_obj_cls = []
        self.sample_2d_pt_num = 1000
        self.pt_num_mesh_small = 500
        self.pt_num_mesh_large = 2600
        self.num_obj = 20

        self.prefix = "data"
        self.use_labelmask = True
        self.debug_mode = False

        self.obj_radius = 1.0
        self.raw_prim_groups = None
    
    def _init_data_info(self):
        # set data list, store in local path
        if self.mode == 'train':
            self.data_list_path = self.Trainlist
        elif self.mode == 'test':
            self.data_list_path = self.Testlist
        else:
            raise NotImplementedError
        self.class_path = self.Classlist
        
        # read data information from files
        with open(self.data_list_path, 'r') as f:
            self.all_data_dirs = f.read().splitlines()
            self.real_data_dirs = [d for d in self.all_data_dirs if d.startswith('scene')] # real-world data path list
            self.syn_data_dirs = [d for d in self.all_data_dirs if d.startswith('synthetic/data')] # synthetic data path list
        with open(self.class_path, 'r') as f:
            import pandas as pd
            self.classes = pd.read_csv(self.class_path, index_col='id')
            
    def load_points(self, root, classes):
        ''' load points of each model from points.xyz and save as dict(key: class id, value: points array) '''
        model_points = {} # {1:[[x1, y1, z1], [x2, y2, z2], ...], 2:[...], 3:[...], ...}
        for idx, cls in classes.iterrows():
            cls_filepath = os.path.join(root, 'objects', cls['name'], 'points.xyz')
            if os.path.isfile(cls_filepath):
                model_points[idx] = np.loadtxt(cls_filepath) # (2621, 3) float
            else:
                print(f"[Warning] {cls} doesn't exist, load model points for {cls} failed")
        return model_points
    
    def get_object_num(self):
        return self.num_obj
    
    def get_sym_list(self):
        return self.symmetry_obj_cls
    
    def get_num_points_mesh(self):
        return self.pt_num_mesh_small

        

######################## dataset utils #############################

def get_discrete_width_bbox(label, border_list, img_w, img_h):
    rows = np.any(label, axis=1)
    cols = np.any(label, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmax += 1
    cmax += 1
    r_b = border_list[binary_search(border_list, rmax - rmin)]
    c_b = border_list[binary_search(border_list, cmax - cmin)]
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_h:
        delt = rmax - img_h
        rmax = img_h
        rmin -= delt
    if cmax > img_w:
        delt = cmax - img_w
        cmax = img_w
        cmin -= delt
    return rmin, rmax, cmin, cmax

def discretize_bbox(rmin, rmax, cmin, cmax, border_list, img_w, img_h):
    rmax += 1
    cmax += 1
    r_b = border_list[binary_search(border_list, rmax - rmin)]
    c_b = border_list[binary_search(border_list, cmax - cmin)]
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_h:
        delt = rmax - img_h
        rmax = img_h
        rmin -= delt
    if cmax > img_w:
        delt = cmax - img_w
        cmax = img_w
        cmin -= delt
    return rmin, rmax, cmin, cmax
    
def binary_search(sorted_list, target):
    l = 0
    r = len(sorted_list)-1
    while l!=r:
        mid = (l+r)>>1
        if sorted_list[mid] > target:
            r = mid
        elif sorted_list[mid] < target:
            l = mid + 1
        else:
            return mid
    return l


if __name__ == "__main__":
    dataset = DTTDDataset(root="./DTTD_IPhone_Dataset/root", mode="test", config_path="./dataset_config")
    dt = dataset[2]
    print("rgb shape: ", dt["rgb"].size())
    print("xyzmap shape: ", dt["xyz"].size())