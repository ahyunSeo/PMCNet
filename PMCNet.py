import os
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import utils
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.module import build_decoder, build_aspp, build_backbone

    
class SymmetryKernelLearning(nn.Module):
    def __init__(self, nangle, is_final=False, sync_bn=True, use_selfsim=True, use_sym_region=True):
        super(SymmetryKernelLearning, self).__init__()

        self.use_selfsim = use_selfsim
        self.use_sym_region = use_sym_region

        ### Build axis-aware kernel masking
        ones = torch.ones((nangle))
        kernel_list = []
        for i in range(nangle):
            if i == 0:
                kernel = torch.diag(ones)
            else:
                kernel = torch.diag(ones[:nangle - i], i) + torch.diag(ones[:i], -(nangle - i))

            kernel_list.append(torch.fliplr(kernel))
        self.kernel = torch.stack((kernel_list), 0)

        self.learnable_kernel = torch.Tensor(nangle, nangle, nangle)
        torch.nn.init.orthogonal_(self.learnable_kernel)
        self.learnable_kernel = nn.Parameter(self.learnable_kernel, requires_grad=True)

        self.kernel = nn.Parameter(self.kernel, requires_grad=False)
    
        self.is_final = is_final
        self.nangle = nangle
        self.sigmoid = nn.Sigmoid()

    def forward(self, corr, half=False):
        ### (b, sym_n, sym_n, des_n, des_n, h, w) corr
        ### (b, n, n) kernel
        kernel = self.kernel * self.learnable_kernel

        if half:
            sym_sim = torch.einsum('bophw, nop -> bnhw', corr, kernel)
            return sym_sim
            
        sym_sim = torch.einsum('bopqrhw, nqr -> bnophw', corr, kernel)
        sym_sim = torch.einsum('bnophw, nop -> bnhw', sym_sim, kernel)
        return sym_sim
 
class SymmetryDetection(nn.Module):
    r"""Symmetry Detection framework"""
    def __init__(self, args, residual_score=False):
        r"""Constructor for Symmetry Detection framework"""
        super(SymmetryDetection, self).__init__()
        self.residual_score = residual_score
        self.process_args(args)
        self.kernel = SymmetryKernelLearning(self.nangle, self.is_final, self.sync_bn, self.use_selfsim, self.use_sym_region, )

    def process_args(self, args):
        self.args = args
        self.sync_bn = args.sync_bn
        self.dsc_size_polar = (int)(args.dsc_size / 2)
        self.sym_size_polar = (int)(args.sym_size / 2)
        self.angle_interval = args.angle_interval
        self.nangle = (int)(360 / self.angle_interval)
        self.use_selfsim = args.use_selfsim
        self.use_sym_region = args.use_sym_region
        self.dsc_ray_length = args.dsc_ray_length
        self.sym_ray_length = args.sym_ray_length

        if self.residual_score:
            # set this sym_det to use conv feature not selfsim
            self.use_selfsim = False
            self.use_sym_region = True

        self.is_final = self.use_sym_region and self.use_selfsim

    def feature_transformation(self, feature, not_norm=False):
        ### To transform the base feature (conv) -> polar self-similarity descriptor
        b, c, h, w = feature.size()
        dsc_grid = self.make_sampling_grid(b, h, w, self.dsc_ray_length, self.dsc_size_polar).to(feature.device)
        patch_feature = F.grid_sample(feature, dsc_grid, padding_mode='border').view(b, c, self.nangle * self.dsc_size_polar, h, w)
        
        feature = F.normalize(feature, p=2, dim=1)
        patch_feature = F.normalize(patch_feature, p=2, dim=1)
        ss_feature = torch.sum(patch_feature * feature.unsqueeze(2), dim=1)
            
        ss_feature = F.relu(ss_feature, inplace=True)
        ss_feature = F.normalize(ss_feature, p=2, dim=1)
            
        return ss_feature
    
    def symmetry_region_feature_map_extraction(self, feature):
        ### To sample with polar sampling grid. Used for the polar region/self-similarity descriptor
        b, c, h, w = feature.size()
        sym_grid = self.make_sampling_grid(b, h, w, self.sym_ray_length, self.sym_size_polar).to(feature.device)

        sym_feature = F.grid_sample(feature, sym_grid, padding_mode='border')
        if self.use_selfsim:
            sym_feature = sym_feature.view(b, self.nangle, self.dsc_size_polar, self.nangle, self.sym_size_polar, h, w)
        else:
            sym_feature = sym_feature.view(b, -1, self.nangle, self.sym_size_polar, h, w)
                
        return sym_feature
    
    def make_sampling_grid(self, b, h, w, length, polar_size):
        if isinstance(polar_size, int):
            length = torch.ones([self.nangle]) * length
        assert length.shape == (self.nangle,)

        grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w)) # HxWx2
        grid = torch.stack([grid_x, grid_y], dim=2).view(1, -1, 2).expand([self.nangle * polar_size, -1, -1]).clone() #(dsc_size_polar)x(HxW)x2
        ratio = torch.linspace(1 / polar_size, 1, polar_size).view([1, polar_size, 1])
        
        length = length.view([self.nangle, 1, 1])
        ang = torch.linspace(0, 360 - self.angle_interval, self.nangle)
        cosd = length * torch.cos(np.pi * ang / 180).view([self.nangle, 1, 1])
        sind = length * torch.sin(np.pi * ang / 180).view([self.nangle, 1, 1])
        
        offset = torch.stack((sind * ratio, - cosd * ratio), dim=3).flatten(0, 1)
        grid = grid.float() + offset
        grid[:, :, 0] = 2 * (grid[:, :, 0].clamp(0, w-1)) - (w-1)
        grid[:, :, 1] = 2 * (grid[:, :, 1].clamp(0, h-1)) - (h-1)
        grid = torch.stack((grid[:, :, 0] / (w - 1), grid[:, :, 1] / (h - 1)), dim=2)
        grid = grid.unsqueeze(0).expand(b, self.nangle * polar_size, -1, 2)
        return grid
    
    def self_correlation(self, feature, axis=1):
        ### use_sym_region : To use polar region descriptor
        ### use_selfsim : To use polar self-similarity descriptor
        if self.use_sym_region:
            if self.use_selfsim:
                feature_corr = torch.einsum('bocqdhw, bpcrdhw -> bopqrcdhw', feature, feature)
                sum_list = [5, 6]
            else:
                # feature : conv
                feature = F.normalize(feature, p=2, dim=1)
                feature_corr = torch.einsum('bcqdhw, bcrdhw -> bqrdhw', feature, feature)
                sum_list = [3]
        else:
            # feature : polar selfsim
            feature_corr = torch.einsum('bochw, bpchw -> bopchw', feature, feature)
            sum_list = [3]

        return feature_corr.sum(sum_list)

    def forward(self, feature, shared_feature=None):
        ### [Step 1]: feature transformation for (conv, selfsim[polar/cartesian])
        if self.use_selfsim:
            feature = self.feature_transformation(feature) # (tensor P)
        
        ### [Step 2]: symmetry region feature map extraction (tensor Z)
        if self.use_sym_region:
            sym_region_feature = self.symmetry_region_feature_map_extraction(feature)
        else:
            b, c, h, w = feature.size()
            sym_region_feature = feature.view(b, self.nangle, self.dsc_size_polar, h, w)
        
        ### [Step 3]: correlation tensor (tensor C)
        sym_region_corr = self.self_correlation(sym_region_feature)

        ### [Step 4]: symmetry kernel learning (tensor S)
        sym_activation = self.kernel(sym_region_corr, half=(not self.use_selfsim or not self.use_sym_region))

        ### [Step 5]: aggregate maximum sym activation (tensor S', S)
        sym_max = torch.softmax(sym_activation, dim=1)
        sym_max, _ = torch.max(sym_max, dim=1, keepdim=True)
        sym_activation = torch.cat((sym_max, sym_activation), dim=1)
    
        return feature, sym_activation
    
class SymmetryDetectionNetwork(nn.Module):
    def __init__(self, args=None):
        super(SymmetryDetectionNetwork, self).__init__()
        self.sync_bn = args.sync_bn
        self.backbone_model = 'resnet'
        self.output_stride = 8
        self.freeze_bn = False
        self.num_classes = 1
        self.angle_interval = args.angle_interval
        self.n_angle = (int)(360 / self.angle_interval)

        if self.sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        use_bn = True
        featdim = 256
        last_convout = featdim
        last_conv_only = True
        
        self.sym_det = SymmetryDetection(args)
        self.res_sym_det = SymmetryDetection(args, residual_score=True)
            
        score_dim = 1 + self.sym_det.nangle
        score_dim *= 2

        last_convin = featdim + score_dim
        
        self.decoder_axis = build_decoder(self.num_classes, self.backbone_model, BatchNorm, last_conv_only, last_convin, last_convout, use_bn=use_bn)
        self.backbone = build_backbone(self.backbone_model, self.output_stride, BatchNorm)
        self.aspp = build_aspp(self.backbone_model, self.output_stride, BatchNorm)
        self.fixed_parameters = self.backbone.parameters()
        if not args.unfreeze:
            for param in self.fixed_parameters:
                param.requires_grad = False
        self.sigmoid = nn.Sigmoid()

    def forward(self, img, lbl, mask, is_syn, a_lbl=None):
        ### [Step 1]: Base feature extraction with (ENC)
        feat, low_level_feat = self.backbone(img)
        feat = self.aspp(feat)

        ### [Step 2-1]: Main score branch (PMC_P)
        _, x = self.sym_det(feat)

        ### [Step 2-2]: Residual score branch (PMC_F)
        _, res_x = self.res_sym_det(feat)

        ### [Step 3-1]: For default, we use two scores altogether
        x = torch.cat((x, res_x), dim=1)        

        ### [Step 3-2]: Final prediction with DEC
        out = self.decoder_axis(x, feat)
        axis_out = F.interpolate(out, size=lbl.size()[2:], mode='bilinear', align_corners=True)

        axis_loss = utils.sigmoid_focal_loss(axis_out, lbl, alpha=0.95)
        axis_out = self.sigmoid(axis_out) # vis purpose
        
        loss = axis_loss
        losses = (axis_loss, axis_loss)

        return axis_out, x, loss, losses
