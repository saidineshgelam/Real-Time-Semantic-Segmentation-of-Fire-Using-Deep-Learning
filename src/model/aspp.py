from typing import Tuple
import torch
from torch import nn
import torch.nn.functional as F

from .deep_cnn import DCNN

class ASPP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        channel_input = 160
        channel_out = 256
        self.intializer(channel_input, channel_out)

    
    def intializer(self,channel_input, channel_out):

        self.cnn = nn.Sequential(
            nn.Conv2d(channel_input, channel_out, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(channel_out, track_running_stats=False),
            nn.ReLU(),
        )

        self.aspp1 = nn.Sequential(
            nn.Conv2d(channel_input, channel_out, kernel_size=3, stride=1,
                      bias=False, dilation=6, padding=6),
            nn.BatchNorm2d(channel_out, track_running_stats=False),
            nn.ReLU(),
        )
        self.aspp2 = nn.Sequential(
            nn.Conv2d(channel_input, channel_out, kernel_size=3, stride=1,
                      bias=False, dilation=12, padding=12),
            nn.BatchNorm2d(channel_out, track_running_stats=False),
            nn.ReLU(),
        )
        self.aspp3 = nn.Sequential(
            nn.Conv2d(channel_input, channel_out, kernel_size=3, stride=1,
                      bias=False, dilation=18, padding=18),
            nn.BatchNorm2d(channel_out, track_running_stats=False),
            nn.ReLU(),
        )

        self.average_pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Conv2d(channel_input, channel_out, kernel_size=1, stride=1,
                      bias=False),
            nn.BatchNorm2d(channel_out, track_running_stats=False),
            nn.ReLU(),
        )

        self.final_convolution = nn.Sequential(
            nn.Conv2d(channel_out*5, channel_out, kernel_size=1, stride=1,
                      bias=False),
            nn.BatchNorm2d(channel_out, track_running_stats=False),
            nn.ReLU(),
            nn.Dropout(0.1),
        )


    def forward(self, layer):
        result1 = self.cnn(layer)
        result2 = self.aspp1(layer)
        result3 = self.aspp2(layer)
        result4 = self.aspp3(layer)
        result5 = self.average_pooling(layer)
        result5 = F.interpolate(result5, size=layer.shape[-2:], mode='bilinear',
                             align_corners=False)
        result = torch.cat([result1, result2, result3, result4, result5], dim=1)
        final_result = self.final_convolution(result)

        return final_result 