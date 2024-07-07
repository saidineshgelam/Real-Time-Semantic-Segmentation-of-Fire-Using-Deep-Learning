import torch
from torch import nn
import torch.nn.functional as F

class MIoU(nn.Module):
    def __init__(self, smooth: float = 1e-6) :
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets) :
        with torch.no_grad():
            inputs = inputs.softmax(-3).argmax(-3, keepdim=True)

            bg_predictions = ~inputs.bool()
            fg_predictions = inputs.bool()
            inputs = torch.cat([bg_predictions, fg_predictions], dim=-3)

            targets = targets.bool()

            intersection = (inputs & targets).sum((-2, -1))
            union = (inputs | targets).sum((-2, -1))

            iou = (intersection + self.smooth) / (union + self.smooth)
            return torch.mean(iou)