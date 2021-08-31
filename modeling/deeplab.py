import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.module import build_decoder, build_aspp, build_backbone

class DeepLab(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False, use_last_conv=True):
        super(DeepLab, self).__init__()


    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        if not self.use_last_conv:
            return x #(B, 304, 129, 129)
        return x
