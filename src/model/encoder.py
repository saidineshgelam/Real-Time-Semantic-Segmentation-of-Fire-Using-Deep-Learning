from typing import Tuple
import torch
from torch import nn

from .aspp import ASPP
from .deep_cnn import DCNN



class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dcnn = DCNN()
        self.aspp_module = ASPP()

    def forward(self, x):

        features_1, features_2, features_3, result = self.dcnn(x)
        features_4 = self.aspp_module(result)
        return features_1, features_2, features_3, features_4