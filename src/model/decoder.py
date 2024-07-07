from typing import Tuple
import torch
from torch import nn
from .aspp import ASPP
from .deep_cnn import DCNN


class Decoder(nn.Module):
 
    def __init__(self, target_size ):

        super().__init__()
        channel_output = 256
        feature_size = (128, 128)
        self.constructor(channel_output, feature_size,target_size)


    def constructor(self,channel_output, feature_size,target_size):
        self.cnn_features1 = nn.Sequential(
            nn.Conv2d(16, channel_output, kernel_size=1, bias=False),
            nn.BatchNorm2d(channel_output, track_running_stats=False),
            nn.ReLU(),
        )
        self.cnn_features2 = nn.Sequential(
            nn.Conv2d(24, channel_output, kernel_size=1, bias=False),
            nn.BatchNorm2d(channel_output, track_running_stats=False),
            nn.ReLU(),
        )
        self.cnn_features3 = nn.Sequential(
            nn.Conv2d(40, channel_output, kernel_size=1, bias=False),
            nn.BatchNorm2d(channel_output, track_running_stats=False),
            nn.ReLU(),
        )

        self.feature_upsampling1 = nn.Upsample(feature_size, mode='bilinear',
                                       align_corners=False)
        self.feature_upsampling3 = nn.Upsample(feature_size, mode='bilinear',
                                       align_corners=False)
        self.feature_upsampling4 = nn.Upsample(feature_size, mode='bilinear',
                                       align_corners=False)


        self.lass_cnn = nn.Sequential(
            nn.Conv2d(channel_output * 4, 256, kernel_size=3, bias=False),
            nn.BatchNorm2d(256, track_running_stats=False),
            nn.ReLU(),
            nn.Conv2d(256, 2, kernel_size=1, stride=1),
        )

        self.final_upsample = nn.Upsample(target_size, mode='bilinear',
                                          align_corners=False)

    def forward(self, feature1, feature2,feature3, feature4):

        result1 = self.cnn_features1(feature1)
        result2 = self.cnn_features2(feature2)
        result3 = self.cnn_features3(feature3)

        result1 = self.feature_upsampling1(result1)
        result3 = self.feature_upsampling3(result3)
        result4 = self.feature_upsampling4(feature4)

        result = torch.cat([result1, result2, result3, result4], dim=1)

        result = self.lass_cnn(result)
        result = self.final_upsample(result)

        return result
