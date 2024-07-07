from time import time
from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import Resize, InterpolationMode
import numpy as np

from .loss_function import FocalLoss
from .mean_intersection_union import MIoU
from .mean_pixel import MPA
from .utils import Checkpoint
from ..model.model import ModelDef

def image_size_change(images,
                       new_size):
    resize = Resize(new_size, InterpolationMode.NEAREST)
    return resize(images)