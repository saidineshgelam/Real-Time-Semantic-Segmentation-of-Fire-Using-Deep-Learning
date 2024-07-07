from time import time
from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import Resize, InterpolationMode
import numpy as np

from .resize_image_batch import image_size_change

from .loss_function import FocalLoss
from .mean_intersection_union import MIoU
from .mean_pixel import MPA
from .utils import Checkpoint
from ..model.model import ModelDef

def validate(
    model, val_dataloader, device,
    resize_evaluation_shape: Optional[Tuple[int, int]] = None
    ) :

    torch.cuda.empty_cache()
    model.eval()

    val_loss_fn = FocalLoss()
    val_mpa_metric = MPA()
    val_miou_metric = MIoU()

    total_val_loss = 0.
    total_val_mpa = 0.
    total_val_miou = 0.
    total_val_fps = 0.

    with torch.no_grad():
        
        for _, (inputs, targets) in enumerate(val_dataloader):
            inputs = inputs.to(device=device)
            targets = targets.to(device=device)

            torch.cuda.synchronize(device)
            start_time = time()
            predictions = model(inputs)
            torch.cuda.synchronize(device)
            duration = time() - start_time

            if resize_evaluation_shape is not None:
                targets = image_size_change(targets, new_size=resize_evaluation_shape)
                predictions = image_size_change(predictions, new_size=resize_evaluation_shape)

            loss = val_loss_fn(predictions, targets)
            total_val_loss += loss.item()

            mpa = val_mpa_metric(predictions, targets)
            miou = val_miou_metric(predictions, targets)

            total_val_mpa += mpa.item()
            total_val_miou += miou.item()
            total_val_fps += inputs.shape[0] / duration

    torch.cuda.empty_cache()
    avg_val_loss = total_val_loss / len(val_dataloader)
    avg_val_mpa = total_val_mpa / len(val_dataloader)
    avg_val_miou = total_val_miou / len(val_dataloader)
    avg_fps = total_val_fps / len(val_dataloader)

    return avg_val_loss, avg_val_mpa, avg_val_miou, avg_fps