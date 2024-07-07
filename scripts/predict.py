import sys
sys.path.append('.')

import argparse
import os
import numpy as np
import torch
from torch import cuda
from torchvision.transforms import ToTensor, Normalize

from src.data.dataset_handler import load_image_from_zip_by_index
from src.model.model import ModelDef
from src.training.utils import Checkpoint
from src.prediction.analysis import plot_image_prediction
import time


def main():
    val = argparse.ArgumentParser(
        description='Script for validating the results of the fire detection '
        'segmentation model.')

    val.add_argument(
        '--checkpoint-file-path','-ckpt', metavar='Checkpoint file path',
        type=str, default=os.path.join('model', 'checkpoints.pth'),
        help='The path of the file where the model checkpoints are loaded.',
        nargs='?', required=False)

    val.add_argument(
        '--train-mean-std-file-path','-ms', metavar='Mean and std file path',
        type=str, default=os.path.join('model', 'mean-std.npy'),
        help='The file path where the train mean and standard deviation are '
        'loaded', nargs='?', required=False)

    val.add_argument(
        '--device', '-d', type=str, default=None, nargs='?',
        help='The device to use for training. If not provided, it is set '
            'automatically.', required=False)

    # Get the arguments.
    arguments = val.parse_args()

    chekpoint_file_path = arguments.checkpoint_file_path
    train_mean_std_file_path = arguments.train_mean_std_file_path
    device = arguments.device

    # Set the original shape.
    image_shape = (3840, 2160)
    # Set the resize shape.
    resized_image_shape = (512, 512)
    # Set the device.
    if device is None:
        device = 'cuda' if cuda.is_available() else 'cpu'

    # Hardcoded image zip file path
    images_zip_path = '/home/kiran/final project/backup/data/Images.zip'

    # Get the total number of images in the zip file
    total_images = len(load_image_from_zip_by_index(images_zip_path, resize_shape=resized_image_shape, image_index=0))

    results = []
    for image_number in range(total_images-1):
        image = load_image_from_zip_by_index(
            images_zip_path, resize_shape=resized_image_shape, image_index=image_number)

        model = ModelDef(resized_image_shape, device=device)
        # Load the best weights of the model.
        checkpoint = Checkpoint(chekpoint_file_path)
        checkpoint.load_best_weights(model)
        model.eval()

        train_mean, train_std = np.load(train_mean_std_file_path)
        
        to_tensor = ToTensor()
        normalize = Normalize(mean=train_mean, std=train_std)
        image_tensor = to_tensor(image)
        image_tensor = normalize(image_tensor)

        print('Starting prediction...')
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(device)
        image_tensor = torch.cat((image_tensor, image_tensor), dim=0)
        with torch.no_grad():
            predicted_mask = model(image_tensor)
        predicted_mask = predicted_mask[0]
        predicted_mask = predicted_mask.softmax(-3).argmax(-3)
        predicted_mask = predicted_mask.cpu().numpy()

        results.append((image, predicted_mask))
        plot_image_prediction(image, predicted_mask, resize_shape=image_shape, save_path=f'/home/kiran/final project/backup/data/results/result_{image_number}.png')
        time.sleep(0.1)

  

if __name__ == '__main__':
    main()  
