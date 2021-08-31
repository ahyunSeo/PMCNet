"""MSCOCO Semantic Segmentation pretraining for VOC."""
# https://github.com/Tramac/awesome-semantic-segmentation-pytorch/blob/afef72dd6630cd6660d82cbb45beb8e57aa49bf9/core/data/dataloader/mscoco.py#L46
import os
import pickle
import torch
import pdb
import random
import numpy as np
import cv2
import scipy.io as io

import json
from tqdm import tqdm
from torch.utils.data import Dataset
from tqdm import trange
import PIL
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO


from utils import *

# dataset: SYM_NYU, SYM_LDRS, SYNTHETIC_COCO, CVPR2013
# N: 'SYM_NYU', L: 'SYM_LDRS', S: 'SYNTHETIC_COCO', D: 'SYM_SDRW'

class SymmetryDatasets(Dataset):
    def __init__(self, root='./sym_datasets', dataset=['SYM_LDRS'], split='train', resize=False, input_size=[513, 513], angle_interval=45, **kwargs):
        super(SymmetryDatasets, self).__init__()
        self.split = split
        self.resize = resize
        self.input_size = input_size
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.angle_interval = angle_interval
        self.n_angle = (int)(360 / self.angle_interval)
        
        self.length = 0 
        self.dataset_names = ['SYM_NYU', 'SYM_LDRS', 'SYNTHETIC_COCO', 'SYM_SDRW']
        self.datasets = []
        for name in dataset:
            if name not in self.dataset_names: continue

            if 'SYN' in name:
                data_set_cls = SYNTHETIC_COCO(split)
            else:
                data_set_cls = RealSymmetryDatasets(split, name)

            print(name, data_set_cls.__len__())
            self.length += data_set_cls.__len__()
            self.datasets.append((name, data_set_cls))
            
    def __getitem__(self, index):
        for name, dataset in self.datasets:
            if index >= dataset.__len__():
                index -= dataset.__len__()
                continue
            if 'SYN' in name:
                img, axis, mask = dataset.__getitem__(index)
                is_syn = True
            else:
                img, axis, axis_lbl, axis_coords = dataset.__getitem__(index)
                is_syn = False
            break

        axis_gs = cv2.GaussianBlur(axis, (5,5), cv2.BORDER_DEFAULT)

        if self.split == 'test' or self.split =='val':
            transform = A.Compose(
                        [ A.Normalize(self.mean, self.std),
                          ToTensorV2(),
                        ], additional_targets={'axis': 'mask', 'axis_gs': 'mask'})
        else:
            transform = A.Compose(
                    [ A.Resize(height=self.input_size[0], width=self.input_size[1]),
                    A.RandomRotate90(),
                    A.Rotate(limit = 15, border_mode = cv2.BORDER_CONSTANT),
                    #A.RandomCrop(height=self.input_size[0], width=self.input_size[1]),
                    A.ColorJitter (brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.5),
                    A.Normalize(self.mean, self.std),
                    ToTensorV2(),
                    ], additional_targets={'axis': 'mask', 'axis_gs': 'mask', 'axis_lbl': 'mask', 'axis_coords': 'bboxes'})
        
        
        if is_syn:
            t = transform(image = img, mask = mask, axis = axis, axis_gs = axis_gs)
            img, mask, axis, axis_gs = t["image"], t["mask"], t["axis"], t["axis_gs"]
        else:
            if self.split == 'test' or self.split =='val':
                t_resize = A.Compose([A.Resize(height=self.input_size[0], width=self.input_size[1])])
                img = t_resize(image = img)["image"]
            t = transform(image = img, axis = axis, axis_gs = axis_gs, axis_lbl=axis_lbl, axis_coords=axis_coords)
            img, axis, axis_gs, axis_lbl, axis_coords = t["image"], t["axis"], t["axis_gs"], t["axis_lbl"], t["axis_coords"]
            mask = torch.zeros_like(axis)
        
        mask = mask.unsqueeze(0)
        axis = axis.unsqueeze(0)
        axis_gs = axis_gs.unsqueeze(0)
        axis_gs = axis_gs / (axis_gs.max() + 1e-5)

        return img, mask, axis, axis_gs, is_syn
        
    def __len__(self):
        return self.length
    
class RealSymmetryDatasets(Dataset):
    def __init__(self, split, dataset, root='./sym_datasets'):
        super(RealSymmetryDatasets, self).__init__()
        self.root = root
        self.get_data_list(dataset, split)

    def get_data_list(self, dataset, split):
        if dataset in ['SYM_NYU', 'NYU']:
            self.img_list, self.gt_list = self.nyu_get_data_list()
        elif dataset in ['SYM_LDRS', 'LDRS']:
            self.img_list, self.gt_list = self.ldrs_get_data_list(split)
        elif dataset in ['SYM_SDRW', 'sdrw']:
            self.img_list, self.gt_list = self.sdrw_get_data_list(split)
        
    def nyu_load_gt(self, path):
        coords = io.loadmat(path)['segments'][0]
        coords = [coord.ravel() for coord in coords]
        return coords
    
    def nyu_get_data_list(self):
        single_path = os.path.join(self.root, 'NYU', 'S')
        multi_path = os.path.join(self.root, 'NYU', 'M')
        single_names = ['I%.3d' % i for i in range(1, 176 + 1)]
        multi_names = ['I0%.2d' % i for i in range(1, 63 + 1)]
        single_img_list = [os.path.join(single_path, single + '.png') for single in single_names]
        multi_img_list = [os.path.join(multi_path, multi + '.png') for multi in multi_names]
        single_gt_list = [self.nyu_load_gt(os.path.join(single_path, single + '.mat')) for single in single_names]
        multi_gt_list = [self.nyu_load_gt(os.path.join(multi_path, multi + '.mat')) for multi in multi_names]

        dataset = {}
        for i, (img, gt) in enumerate(zip(single_img_list + multi_img_list, single_gt_list + multi_gt_list)):
            dataset[img] = gt
                
        img_list = list(dataset.keys())
        gt_list = list(dataset.values())
        
        return img_list, gt_list

    def ldrs_get_data_list(self, split):
        LDRS_path = os.path.join(self.root, 'LDRS', split)
        files = os.listdir(LDRS_path)
        json_files = list(filter(lambda x: x.find('.json') != -1, files))
        
        img_list = []
        gt_list = []
        
        for file in json_files:
            json_path = os.path.join(LDRS_path, file)
            img_path = json_path.rstrip('.json') + '.jpg'
            with open(json_path) as json_file:
                json_data = json.load(json_file)
                gt_list.append([(axis['points'][0][0], axis['points'][0][1], axis['points'][1][0], axis['points'][1][1]) for axis in json_data['shapes']])
            img_list.append(img_path)
        return img_list, gt_list

    def sdrw_get_data_list(self, split):
        if split in ['test', 'val']:
            cvpr_path = os.path.join(self.root, 'SDRW')
            single_path = os.path.join(cvpr_path, 'reflection_testing', 'single_testing')
            multi_path = os.path.join(cvpr_path, 'reflection_testing', 'multiple_testing')
            cvpr_gt = io.loadmat(os.path.join(cvpr_path, 'reflectionTestGt2013.mat'))
            single_gt = cvpr_gt['gtS']
            multi_gt = cvpr_gt['gtM']

            single_img_list = [os.path.join(single_path, single[0][0].strip()) for single in single_gt[0][0][0]]
            single_img_list[5] = single_img_list[5][:-4] + '.png'
            single_gt_list = [single for single in single_gt[0][0][1]]
            multi_img_list = [os.path.join(multi_path, multi[0][0]) for multi in multi_gt[0][0][0]]
            multi_gt_list = [multi for multi in multi_gt[0][0][1]]
            multi_img_list[0] = multi_img_list[0][:-4] + '.png'
            multi_img_list[1] = multi_img_list[1][:-4] + '.png'
            
            dataset = {}

            for i, (img, gt) in enumerate(zip(single_img_list + multi_img_list, single_gt_list + multi_gt_list)):
                if img in dataset: 
                    dataset[img].append(gt)
                else:
                    dataset[img] = [gt]
            
        else:
            cvpr_path = os.path.join(self.root, 'SDRW', 'reflection_training')
            single_path = os.path.join(cvpr_path, 'single_training')
            multi_path = os.path.join(cvpr_path, 'multiple_training')
            single_gt = io.loadmat(os.path.join(cvpr_path, 'singleGT_training', 'singleGT_training.mat'))['gt']
            multi_gt = io.loadmat(os.path.join(cvpr_path, 'multipleGT_training', 'multipleGT_training.mat'))['gt']

            single_img_list = [os.path.join(single_path, single[0][0].strip()) for single in single_gt['name']]
            multi_img_list = [os.path.join(multi_path, multi[0][0].strip()) for multi in multi_gt['name']]
            single_gt_list = [single[0][0] for single in single_gt['ax']]
            multi_gt_list = [multi[0] for multi in multi_gt['ax']]

            dataset = {}

            for img, gt in zip(single_img_list, single_gt_list):
                if img in dataset: 
                    dataset[img].append(gt)
                else:
                    dataset[img] = [gt]

            for img, gt in zip(multi_img_list, multi_gt_list):
                for _gt in gt:
                    if img in dataset:
                        dataset[img].append(_gt)
                    else:
                        dataset[img] = [_gt]
                
        img_list = list(dataset.keys())
        gt_list = list(dataset.values())
        
        return img_list, gt_list
    
    def __getitem__ (self, index):
        img = self.img_list[index]
        img = Image.open(img).convert('RGB')
        axis = self.gt_list[index]
        axis, line_coords = draw_axis(axis, img.size)
        img = match_input_type(img)
        _axis = (axis > 0).astype(axis.dtype)
        return img, _axis, axis, line_coords
    
    def __len__(self):
        return len(self.img_list)

class SYNTHETIC_COCO(Dataset):
    def __init__(self, split, root='./coco_data', year='2014'):
        super(SYNTHETIC_COCO, self).__init__()
        
        self.root = root
        self.nangle = 2
        self.ninst = 10
        self.data_list = self.get_data_list(split, year)
        
    def get_data_list(self, split, year):
        ann_file = os.path.join(self.root, 'annotations', f"instances_{split}{year}.json")
        ids_file = os.path.join(self.root, 'annotations', f"{split}_ids.mx")
        self.root = os.path.join(self.root, f"{split}{year}")
        self.coco = COCO(ann_file)
        self.ids_all = list(self.coco.imgs.keys())
        if os.path.exists(ids_file):
            with open(ids_file, 'rb') as f:
                ids = pickle.load(f)
        else:
            ids = self._preprocess(self.ids_all, ids_file)
        
        return ids
    
    def get_rot_matrices (self, h, w):
        center = (w/2, h/2)
        rand_ang = np.random.randint(180 / self.nangle, size = self.nangle)
        angles = np.linspace(start = -90, stop = 90, num = self.nangle, endpoint= False) + rand_ang
        rot_m = []
        unrot_m = []
        for angle in angles:
            m = cv2.getRotationMatrix2D(center, angle, 1.0)
            un_m = cv2.getRotationMatrix2D(center, -angle, 1.0)
            rot_m.append(m)
            unrot_m.append(un_m)
        rot_m = np.stack(rot_m, axis=0) # ang x 2 x 3
        unrot_m = np.stack(unrot_m, axis=0) # ang x 2 x 3
        return rot_m, unrot_m, angles
        
    def get_rot_anns(self, ann, h, w):
        rot_ann = []
        
        ann = sorted(ann, key=lambda x: -x['area'])
        for idx, instance in enumerate(ann):
            rot_m, unrot_m, angles = self.get_rot_matrices(h, w)
            rot_instances = []
            xmin, ymin = 1e3 * np.ones(self.nangle), 1e3 * np.ones(self.nangle)
            xmax, ymax = np.zeros(self.nangle), np.zeros(self.nangle)
            
            ## 'iscrowd' option change 'segmentation' in other formats, thus bypass the instance having this option
            if instance['iscrowd']: 
                continue
            for seg in instance['segmentation']:
                ## rotate contour points for all angles.
                points = np.asarray(seg).reshape(-1, 2)
                ones = np.ones(shape=(len(points), 1))
                points_ones = np.hstack([points, ones])
                rot_points = np.matmul(rot_m, points_ones.T).transpose(1, 2, 0) # (x,y) x points x ang
                
                ## limit boundaries
                rot_points = np.where(rot_points < 0, 0, rot_points)
                rot_points[0] = np.where(rot_points[0] >= w - 1, w - 1, rot_points[0])
                rot_points[1] = np.where(rot_points[1] >= h - 1, h - 1, rot_points[1])
            
                ## integrate boundaries into one segment
                xmin_inst, xmax_inst, ymin_inst, ymax_inst = np.min(rot_points[0], 0), np.max(rot_points[0], 0), np.min(rot_points[1], 0), np.max(rot_points[1], 0)
                xmin, ymin = np.minimum(xmin, xmin_inst), np.minimum(ymin, ymin_inst)
                xmax, ymax = np.maximum(xmax, xmax_inst), np.maximum(ymax, ymax_inst)
                rot_instances.append(rot_points.transpose(2, 1, 0))
            
            ## filtering & set cut axis by random
            avail_ang = (xmax - xmin + 1) * (ymax - ymin)> h*w/16
            cut_axis = (xmin + np.random.uniform(1/3, 2/3) * (xmax - xmin)).astype(int)
            for i in np.argwhere(avail_ang):
                i = i[0]
                ## flip points that are left from cut axis, but right from the limitx.
                flip_instances = []
                limitx = cut_axis[i] - (w - cut_axis[i])
                for rot_points in rot_instances:
                    npoints = len(rot_points[i,:,0])
                    is_left = rot_points[i,:,0] < cut_axis[i]
                    ind_left = np.where(np.diff(np.where(is_left)[0]) > 1)[0] + 1
                    left_points = np.split(rot_points[i,is_left,:], ind_left, axis=0)
                    
                    if len(left_points[0]) == 0 or len(left_points[0]) == npoints: continue
                    if (is_left[0] == True) and (is_left[-1] == True):
                        left_points[0] = np.concatenate((left_points[-1], left_points[0]), axis=0)
                        del left_points[-1]
                    for lpoints in left_points:
                        lpoints[:,0] = np.where(lpoints[:,0]<limitx, limitx, lpoints[:,0])
                        right_points = np.stack((lpoints[:,0] + 2 * (cut_axis[i] - lpoints[:,0]), lpoints[:,1]), axis=1)
                        flip_points = np.concatenate((lpoints, np.flip(right_points, 0)), axis=0)
                        
                        if len(flip_points) <= 2:
                            continue
                        flip_instances.append(flip_points)
                if len(flip_instances) == 0:
                    continue
                
                ## make coco-style instance after un-rotating flipped instance.
                instance_copy = instance.copy()
                instance_copy['segmentation'] = []
                instance_copy['angle'] = angles[i]
                instance_copy['cutaxis'] = cut_axis[i]
                
                only_flip_pos = unrot_pos = distort_pos = np.ones([4]) * -1e3
                for flip_points in flip_instances:
                    ones = np.ones(shape=(len(flip_points), 1))
                    flip_points_ones = np.hstack([flip_points, ones])
                    unrot_points = np.matmul(unrot_m[i], flip_points_ones.T).T # points x (x,y)
                                        
                    unrot_points = np.where(unrot_points < 0, 0, unrot_points)
                    unrot_points[:, 0] = np.where(unrot_points[:, 0] >= w - 1, w - 1, unrot_points[:, 0])
                    unrot_points[:, 1] = np.where(unrot_points[:, 1] >= h - 1, h - 1, unrot_points[:, 1])
                    
                    flip_points[:, 0] = np.where(flip_points[:, 0] >= w - 1, w - 1, flip_points[:, 0])
                
                    instance_copy['segmentation'].append(unrot_points.flatten())
                    
                    bound_unrot_inst = -np.min(unrot_points[:, 0], 0), np.max(unrot_points[:, 0], 0), -np.min(unrot_points[:, 1], 0), np.max(unrot_points[:, 1], 0) 
                    bound_flip_inst = -np.min(flip_points[:, 0], 0), np.max(flip_points[:, 0], 0), -np.min(flip_points[:, 1], 0), np.max(flip_points[:, 1], 0) 
                    unrot_pos = np.maximum(unrot_pos, bound_unrot_inst)
                    only_flip_pos = np.maximum(only_flip_pos, bound_flip_inst)
                
                unrot_pos = np.ceil(unrot_pos).astype(int)
                only_flip_pos = np.ceil(only_flip_pos).astype(int)
                unrot_bbox = np.stack([-unrot_pos[0], -unrot_pos[2], (unrot_pos[0] + unrot_pos[1] + 1), (unrot_pos[2] + unrot_pos[3] + 1)], 0)
                only_flip_bbox = np.stack([-only_flip_pos[0], -only_flip_pos[2], (only_flip_pos[0] + only_flip_pos[1] + 1), (only_flip_pos[2] + only_flip_pos[3] + 1)], 0)
                instance_copy['bbox'] = unrot_bbox
                instance_copy['only_flip_bbox'] = only_flip_bbox
                rot_ann.append(instance_copy)
            
            if (len(rot_ann) >= self.ninst): break
        
        return rot_ann
    
    def rot_flip_unrot_merge (self, img, sym_ann):
        merged_img = np.asarray(img).copy()
        for seg in sym_ann:
            sym_img = np.asarray(img.rotate(seg['angle'], resample=PIL.Image.BILINEAR)).copy()
            bbox = seg['only_flip_bbox']
            sym_img[bbox[1]:bbox[1]+bbox[3], bbox[0]+bbox[2]-1:seg['cutaxis']:-1] = sym_img[bbox[1]:bbox[1]+bbox[3], seg['cutaxis'] - (bbox[0] + bbox[2] - 1 - seg['cutaxis']):seg['cutaxis']]
            
            sym_img = Image.fromarray(sym_img).rotate(-seg['angle'], resample=PIL.Image.BILINEAR)
            sym_img = np.asarray(sym_img)
            
            bbox = seg['bbox']
            merged_img[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]] = sym_img[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
        return merged_img
    
    def construct_graph (self, rot_ann):
        ## Get maximally occupied region by DFS.
        E = np.zeros((len(rot_ann), len(rot_ann))).astype(bool)
        for i in range(1, len(rot_ann)):
            for j in range(i):
                flag = 0
                if (rot_ann[i]['bbox'][0] + rot_ann[i]['bbox'][2] < rot_ann[j]['bbox'][0]): flag = 1 
                if (rot_ann[i]['bbox'][0] > rot_ann[j]['bbox'][0] + rot_ann[j]['bbox'][2]): flag = 1 
                if (rot_ann[i]['bbox'][1] + rot_ann[i]['bbox'][3] < rot_ann[j]['bbox'][1]): flag = 1 
                if (rot_ann[i]['bbox'][1] > rot_ann[j]['bbox'][1] + rot_ann[j]['bbox'][3]): flag = 1 
                if flag :
                    E[i][j]=True
                    E[j][i]=True
        return E
    
    def get_sym_info (self, rot_ann, h, w, preprocess = False):
        sym_axis = np.zeros((h, w))
        sym_axis_gs = np.zeros((h, w))
        sym_mask =  np.zeros((h, w), dtype=np.float32)
        inst_axis = []
        inst_mask = []
        E = self.construct_graph(rot_ann)
        S = np.zeros((len(rot_ann)))
        
        for i, seg in enumerate(rot_ann):
            rle = coco_mask.frPyObjects(seg['segmentation'], h, w)
            mask = coco_mask.decode(rle)
            
            if len(mask.shape) >= 3:
                mask = ((np.sum(mask, axis=2)) > 0).astype(np.uint8)
                
            axis = np.zeros((h, w))
            bbox = seg['only_flip_bbox']
            axis[bbox[1]:bbox[1]+bbox[3], seg['cutaxis'] : seg['cutaxis'] + 1] = 1.0
            
            axis = np.asarray(Image.fromarray(axis).rotate(-seg['angle'], resample=PIL.Image.NEAREST))
            axis = np.logical_and(axis > 0, mask > 0).astype(np.float)
            axis_length = axis.sum()
            
            if axis_length < 0.8 * bbox[3]: axis_length = -1
            S[i] = axis_length
            
            inst_axis.append(axis)
            inst_mask.append(mask)
                
        dfs = DFS(E, S)
        max_state = dfs.forward()
        
        if preprocess:
            if len(S) == 0:
                return 0, 0
            return dfs.max_score, dfs.max_visited
        else:
            sym_ann = [rot_ann[i] for i in max_state]
            inst_axis = [inst_axis[i] for i in max_state]
            inst_mask = [inst_mask[i] for i in max_state]
            
            for idx, (mask, axis) in enumerate(zip(inst_mask, inst_axis)):
                sym_axis += axis
                sym_mask += mask * (1 + idx)
            return sym_ann, sym_axis, sym_mask
    
    def blend_img(self, fg, bg, mask):
        fg = (fg / 255.0).astype(np.float32)
        bg = (bg / 255.0).astype(np.float32)
        
        mean_fg, std_fg = fg.mean(axis=(0, 1), keepdims=True), fg.std(axis=(0, 1), keepdims=True)
        mean_bg, std_bg = bg.mean(axis=(0, 1), keepdims=True), bg.std(axis=(0, 1), keepdims=True)

        norm_fg = (fg - mean_fg) / std_fg
        norm_fg = (norm_fg * std_bg) + mean_bg

        mask = np.expand_dims((mask > 0), -1).astype(np.float)
        alpha = np.expand_dims(cv2.GaussianBlur(mask, (5, 5), cv2.BORDER_DEFAULT), -1)
        alpha_blend = alpha * (norm_fg) + (1 - alpha) * bg
        alpha_blend = np.clip((alpha_blend * 255.0).astype(np.int32), 0, 255) 
        return alpha_blend.astype(np.uint8)
    
    def resize_bg_as_fg (self, bg, fg_shape):
        self.transform = A.Compose(
                        [ A.SmallestMaxSize(max_size=max(fg_shape)+100),
                          A.RandomCrop(height=fg_shape[0], width=fg_shape[1])])
        t = self.transform(image = bg)
        return t["image"]
    
    def _preprocess(self, ids, ids_file):
        print("Preprocessing mask, this will take a while." + 
              "But don't worry, it only run once for each split.")
        coco = self.coco
        tbar = trange(len(ids))
        new_ids = []
        for i in tbar:
            img_id = ids[i]
            ann = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
            img_metadata = coco.loadImgs(img_id)[0]
            h, w = img_metadata['height'], img_metadata['width']
            
            rot_ann = self.get_rot_anns(ann, h, w)
            sym_length_all, nvisit = self.get_sym_info(rot_ann, h, w, True)
            if sym_length_all >= (h + w) / 2 and nvisit > 2:
                new_ids.append(img_id)
            tbar.set_description('Doing: {}/{}, got {} qualified images'. \
                                 format(i, len(ids), len(new_ids)))
        print('Found number of qualified images: ', len(new_ids))
        with open(ids_file, 'wb') as f:
            pickle.dump(new_ids, f)
        return new_ids
    
    
    def __getitem__(self, index):
        coco = self.coco
        n = len(self.ids_all)
        bg_id = self.ids_all[random.randint(0, n - 1)]
        bg_metadata = coco.loadImgs(bg_id)[0]
        bg_path = bg_metadata['file_name']
        bg = Image.open(os.path.join(self.root, bg_path)).convert('RGB')
        bg = np.asarray(bg)
        
        img_id = self.data_list[index]
        img_metadata = coco.loadImgs(img_id)[0]
        img_path = img_metadata['file_name']
        img = Image.open(os.path.join(self.root, img_path)).convert('RGB')
        h, w = img_metadata['height'], img_metadata['width']
        
        ann = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        rot_ann = self.get_rot_anns(ann, h, w)
        sym_ann, sym_axis, sym_mask = self.get_sym_info(rot_ann, h, w)
        fg = self.rot_flip_unrot_merge(img, sym_ann)
        bg = self.resize_bg_as_fg(bg, fg.shape)
        sym_img = self.blend_img(fg, bg, sym_mask)
        
        return sym_img, sym_axis, sym_mask
    
    def __len__(self):
        return len(self.data_list)

if __name__ == '__main__':
    import torch.utils.data as data
    
    trainset = COCOSegmentation()
    #trainset.__getitem__(17)
    train_data = data.DataLoader(
        trainset, batch_size=16, shuffle=False,
        num_workers=4)
    
    for i, data in enumerate(tqdm(train_data)):
        img, mask, axis = preprocess_batch(data)
        for j in range(len(img)):
             plt.imshow(transforms.ToPILImage()(img[j]))
             plt.show()
             plt.imshow(transforms.ToPILImage()(mask[j]))
             plt.show()
