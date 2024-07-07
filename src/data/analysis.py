from typing import Literal, Optional, Tuple
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from .dataset_handler import resize_images

def show_sample_data(images, masks, num_samples: int = 6, title: Optional[str] = None,resize_shape: Optional[Tuple[int, int]] = None ) :

    len_images = min(len(images), len(masks))

    sample_indices = np.linspace(0, len_images - 1, num=num_samples,
                                 dtype=int)
    _, axes = plt.subplots(3, num_samples, figsize=(30, 8))

    images = images[sample_indices]
    masks = masks[sample_indices]
    
    if resize_shape is not None:
        images = resize_images(images, resize_shape)
        masks = resize_images(masks, resize_shape)
    
    

    for i, (img, mask) in enumerate(zip(images, masks)):
        ax = axes[0, i]
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.axis('off')

        ax = axes[1, i]
        ax.imshow(mask, cmap='gray', vmin=0., vmax=1.)
        ax.axis('off')
        legend_elements = [
            Patch(facecolor='w', edgecolor='black',label='Fire mask')]
        ax.legend(handles=legend_elements, loc='upper left')

        ax = axes[2, i]
        highlighted_roi = get_highlighted_roi_by_mask(
            img, mask, highlight_channel='red')
        ax.imshow(cv2.cvtColor(highlighted_roi, cv2.COLOR_BGR2RGB))
        ax.axis('off')
        legend_elements = [
            Patch(facecolor='r', edgecolor='black', label='Fire ROI')]
        ax.legend(handles=legend_elements, loc='upper left')
    if title is None:
        title = f'Representation of {num_samples} images from the dataset '
        'along with their fire mask and the highlighted segmentation'
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def get_highlighted_roi_by_mask(image, mask,highlight_channel: Literal['blue', 'green', 'red'] = 'green'):
    channel_map = { 'blue': 0, 'green': 1, 'red': 2 }

    mask = mask.astype(np.uint8)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    for c, v in channel_map.items():
        if c != highlight_channel:
            mask[..., v] = 0

    mask[mask == 1.] = 255
    return cv2.addWeighted(mask, 0.9, image, 1, 0)
