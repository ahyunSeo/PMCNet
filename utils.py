import os
import re
from functools import reduce
from operator import add
import pickle
import torch
import random
import inspect
import numpy as np
import json
import PIL
import time
import cv2
import scipy.io as io
from scipy.io import loadmat
from tqdm import tqdm, trange

import torch.nn as nn
from torchvision.models import vgg, resnet
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from PIL import Image, ImageDraw
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO

from bsds.bsds.evaluate_boundaries import * 

"""MSCOCO Semantic Segmentation pretraining for VOC."""
# https://github.com/Tramac/awesome-semantic-segmentation-pytorch/blob/afef72dd6630cd6660d82cbb45beb8e57aa49bf9/core/data/dataloader/mscoco.py#L46

##########################
### Data #################
##########################

class DFS:
    def __init__(self, E, S):
        self.E = E
        self.S = S
        self.visited = np.zeros(len(S)).astype(bool)
        self.state = []
        self.score = 0
        self.max_state = []
        self.max_score = 0
        self.max_visited = 0
        
    def dfs(self):
        if (self.max_score < self.score * self.visited.sum()):
            del self.max_score
            self.max_state = self.state.copy()
            self.max_visited = self.visited.sum()
            self.max_score = self.score * self.max_visited
        for j in range(len(self.S)):
            if self.visited[j]: continue
            flag = 1
            for k in self.state:
                if not self.E[j][k]: flag = 0
            if flag:
                self.visited[j]=True
                self.state.append(j)
                self.score += self.S[j]
                self.dfs()
                self.state.pop()
                self.score -= self.S[j]
                self.visited[j]=False
                
    def forward(self):
        self.dfs()
        return self.max_state
    
def draw_axis(lines, size):
    axis = Image.new('L', size)
    # w, h = img.size
    draw = ImageDraw.Draw(axis)
    length = np.array([size[0], size[1], size[0], size[1]])
    
    # x1, y1, x2, y2
    line_coords = []

    for idx, coords in enumerate(lines):
        if coords[0] > coords[2]:
            coords = np.roll(coords, -2)
        draw.line(list(coords), fill=(idx + 1))
        coords = np.array(coords).astype(np.float32)
        _line_coords = coords / length
        line_coords.append(_line_coords)
    axis = np.asarray(axis).astype(np.float32)
    return axis, line_coords

def match_input_type(img):
    img = np.asarray(img)
    if img.shape[-1] != 3:
        img = np.stack((img, img, img), axis=-1)
    return img 
    
def norm(img):
    mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    img = (img - mean) / std
    return img

def unnorm(img):
    mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    img = img * std + mean
    return img

##########################
### Train ################
##########################

def sigmoid_focal_loss(
    source: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: str = "none",
    is_logits=True
):
    """
    Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py .
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default = 0.25
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    if is_logits:
        p = nn.Sigmoid()(source)
        ce_loss = F.binary_cross_entropy_with_logits(
            source, targets, reduction="none"
        )
    else:
        p = source
        ce_loss = F.binary_cross_entropy(source, targets, reduction="none")
    
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss

def adjust_learning_rate(optimizer, epoch):
    lr = args.lr * (0.1 ** (epoch // (args.num_epochs * 0.5))) * (0.1 ** (epoch // (args.num_epochs * 0.75)))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


##########################
### Evaluation ###########
##########################

class AxisEvaluation(object):
    def __init__(self, n_thresh=100, max_dist=0.01):
        self.count_r_overall = np.zeros((n_thresh,))
        self.sum_r_overall = np.zeros((n_thresh,))
        self.count_p_overall = np.zeros((n_thresh,))
        self.sum_p_overall = np.zeros((n_thresh,))
        self.num_samples = 0
        self.n_thresh = n_thresh
        self.max_dist = max_dist
    
    def f1_score(self, ):
        counts = (self.count_r_overall, self.sum_r_overall, self.count_p_overall, self.sum_p_overall)
        return compute_rec_prec_f1(counts[0], counts[1], counts[2], counts[3])

    def match_and_count(self, pred, gt_b):
        # Expect pred (H, W) gt (n, H, W), n is the # of gt annotations
        count_r, sum_r, count_p, sum_p, _ = \
            multi_evaluate_boundaries(pred, gt_b, thresholds=self.n_thresh, max_dist=self.max_dist)
        
        self.count_r_overall += count_r
        self.sum_r_overall += sum_r
        self.count_p_overall += count_p
        self.sum_p_overall += sum_p
        self.num_samples += 1

    def __call__ (self, pred, gt_b):
        # Evaluate predictions (B, 1, H, W)
        for b in range(pred.shape[0]):
            _pred = pred[b, 0, :, :].cpu().detach().numpy()
            _axis = gt_b[b, :, :, :].cpu().numpy()
            self.match_and_count(_pred, _axis)
