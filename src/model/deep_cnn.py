from torch import nn
import torch.nn.functional as F

from .mobile_net_v3 import MobileNetV2

class DCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn_input = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16, track_running_stats=False),
            nn.Hardswish(),
        )
        self.bottle1_func()
        self.bottle2_func()
        self.bottle3_func()
        self.bottle4_func()

    def bottle1_func(self):
        self.mobile_bottle_neck_1 =  nn.Sequential(MobileNetV2(16, 16, 16, 3, nn.ReLU, False, stride=1))

    def bottle2_func(self):
        self.mobile_bottle_neck_2 =  nn.Sequential(
            MobileNetV2(16, 64, 24, 3, nn.ReLU, False, stride=2,
                             padding=1),
            MobileNetV2(24, 72, 24, 3, nn.ReLU, False, stride=1),
        )

    def bottle3_func(self):
        self.mobile_bottle_neck_3 =  nn.Sequential(
            MobileNetV2(24, 72, 40, 5, nn.ReLU, True, stride=2,
                             padding=2),
            MobileNetV2(40, 120, 40, 5, nn.ReLU, True, stride=1,
                             padding=2),
            MobileNetV2(40, 120, 40, 5, nn.ReLU, True, stride=1,
                             padding=2),
        )

    def bottle4_func(self):
        self.mobile_bottle_neck_4 =  nn.Sequential(
            MobileNetV2(40, 240, 80, 3, nn.Hardswish, False, stride=2),
            MobileNetV2(80, 200, 80, 3, nn.Hardswish, False, stride=1),
            MobileNetV2(80, 184, 80, 3, nn.Hardswish, False, stride=1),
            MobileNetV2(80, 184, 80, 3, nn.Hardswish, False, stride=1),
            MobileNetV2(80, 480, 112, 3, nn.Hardswish, True, stride=1),
            MobileNetV2(112, 672, 160, 3, nn.Hardswish, True, stride=1),
            MobileNetV2(160, 672, 160, 5, nn.Hardswish, True, stride=1,
                             padding=2),
            MobileNetV2(160, 960, 160, 5, nn.Hardswish, True, stride=1,
                             padding=2),
            MobileNetV2(160, 960, 160, 5, nn.Hardswish, True, stride=1,
                             padding=2),
        )


    def forward(self, x):

        result = self.cnn_input(x)
        feature_1 = self.mobile_bottle_neck_1(result)
        feature_2 = self.mobile_bottle_neck_2(feature_1)
        feature_3 = self.mobile_bottle_neck_3(feature_2)
        result = self.mobile_bottle_neck_4(feature_3)

        return feature_1, feature_2, feature_3, result
