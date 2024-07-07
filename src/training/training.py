from time import time
from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import Resize, InterpolationMode
import numpy as np

from .validation import validate

from .loss_function import FocalLoss
from .mean_intersection_union import MIoU
from .mean_pixel import MPA
from .utils import Checkpoint
from ..model.model import ModelDef

def train(
    model, optimizer,
    train_dataloader, val_dataloader,
    epochs, validation_step, device,
    checkpoint: Optional[Checkpoint] = None, lr_schedulers: List[object] = [],
    reload_best_weights: bool = True) :

    criterion = FocalLoss()
    mpa_metric = MPA()
    miou_metric = MIoU()

    step_lr_scheduler = lr_schedulers[0] if len(lr_schedulers) > 0 else None
    plateau_lr_scheduler = lr_schedulers[1] if len(lr_schedulers) > 1 else None

    metrics = ['train_loss', 'train_mpa', 'train_miou', 'val_loss', 'val_mpa',
               'val_miou']
    history = { m: [] for m in metrics }

    model.train()

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')

        torch.cuda.empty_cache()

        running_train_loss = 0.
        running_train_mpa = 0.
        running_train_miou = 0.

        start_time = time()

        for batch_idx, (x, y) in enumerate(train_dataloader):
            batch_steps = batch_idx + 1

            x = x.to(device=device)
            y = y.to(device=device)

            y_pred = model(x)

            loss = criterion(y_pred, y)
            running_train_loss += loss.item()

            optimizer.zero_grad()
            mpa = mpa_metric(y_pred, y)
            miou = miou_metric(y_pred, y)

            running_train_mpa += mpa.item()
            running_train_miou += miou.item()
            loss.backward()

            optimizer.step()

            plateau_lr_scheduler.step(running_train_loss / batch_steps)

            epoch_time = time() - start_time
            batch_time = epoch_time / batch_steps

  
            if batch_steps % validation_step == 0:
                model.eval()

                val_results = validate(model, val_dataloader, device)
                val_loss, val_mpa, val_miou, _ = val_results

                if checkpoint is not None:
                    accuracy_sum = val_mpa + val_miou
                    checkpoint.save_best(model, optimizer, accuracy_sum)

                model.train()

        model.eval()

        train_loss = running_train_loss / len(train_dataloader)
        train_mpa = running_train_mpa / len(train_dataloader)
        train_miou = running_train_miou / len(train_dataloader)

        val_results = validate(model, val_dataloader, device)
        val_loss, val_mpa, val_miou, _ = val_results

        history['train_loss'].append(train_loss)
        history['train_mpa'].append(train_mpa)
        history['train_miou'].append(train_miou)

        history['val_loss'].append(val_loss)
        history['val_mpa'].append(val_mpa)
        history['val_miou'].append(val_miou)

        if checkpoint is not None:
            accuracy_sum = val_mpa + val_miou
            checkpoint.save_best(model, optimizer, accuracy_sum)

        step_lr_scheduler.step()

        model.train()

    if checkpoint is not None and reload_best_weights:
        checkpoint.load_best_weights(model)

    model.eval()

    torch.cuda.empty_cache()

    for k, v in history.items():
        history[k] = np.array(v)

    return history

