import torch
from typing import Tuple
from torch import nn
from .encoder import Encoder
from .decoder import Decoder



class ModelDef(nn.Module):
    def __init__(self, size, run_item):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(target_size=size)
        self.to(run_item)

    def forward(self, input_value):
        feature1, feature2, feature3, feature4 = self.encoder(input_value)
        result = self.decoder(feature1, feature2, feature3, feature4)
        return result
