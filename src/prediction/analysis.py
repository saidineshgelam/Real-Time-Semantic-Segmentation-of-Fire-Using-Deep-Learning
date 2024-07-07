"""Module containing functions to analyze the prediction results."""
from typing import Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import cv2
import time

from src.data.dataset_handler import resize_images
from src.data.analysis import get_highlighted_roi_by_mask







def plot_image_prediction(image, mask,
                          resize_shape: Optional[Tuple[int, int]] = None,
                          save_path: Optional[str] = None
                          ):

    _, axes = plt.subplots(1, 3, figsize=(15, 8))

    if resize_shape is not None:
        [image] = resize_images([image], resize_shape)
        [mask] = resize_images([mask], resize_shape)

    ax = axes[0]
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.axis('off')

    ax = axes[1]
    ax.imshow(mask, cmap='gray', vmin=0., vmax=1.)
    ax.axis('off')
    legend_elements = [
        Patch(facecolor='w', edgecolor='black',label='Fire mask')]
    ax.legend(handles=legend_elements, loc='upper left')

    ax = axes[2]
    highlighted_roi = get_highlighted_roi_by_mask(
        image, mask, highlight_channel='red')
    ax.imshow(cv2.cvtColor(highlighted_roi, cv2.COLOR_BGR2RGB))
    ax.axis('off')
    legend_elements = [
        Patch(facecolor='r', edgecolor='black', label='Fire ROI')]
    ax.legend(handles=legend_elements, loc='upper left')
    title = 'An image along with the predicted fire mask and the highlighted '
    'segmentation'
    plt.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        time.sleep(0.1)
    else:
        plt.show()