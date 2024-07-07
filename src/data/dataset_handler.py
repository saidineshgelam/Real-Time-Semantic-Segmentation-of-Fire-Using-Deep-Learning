import os
import re
from typing import Tuple
from zipfile import ZipFile
import cv2
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm

def load_images_from_zip(zip_file_path: str, resize_shape: Tuple[int, int], are_masks: bool) -> np.ndarray:
    images = []

    with ZipFile(zip_file_path) as zf:
        # Get file names list and skip the first folder name.
        file_names = zf.namelist()[1:]
        # Sort the file names by their image number.
        file_names = sorted(
            file_names,
            key=lambda x: int(re.findall(r'[\d]+', x)[0]))

        for file_name in tqdm(file_names):
            # Read the current file.
            data = zf.read(file_name)
            # Decode the file into a numpy array.
            if are_masks:
                img = cv2.imdecode(np.frombuffer(data, np.uint8),
                                   cv2.IMREAD_GRAYSCALE)                         
            else:
                img = cv2.imdecode(np.frombuffer(data, np.uint8),
                                   cv2.IMREAD_COLOR)
            if img is not None:
                img = cv2.resize(
                    img, resize_shape,interpolation=cv2.INTER_NEAREST)
            images.append(img)

    return np.array(images)

def load_image_from_zip_by_index(
    zip_file_path: str, resize_shape: Tuple[int, int],
    image_index: int) -> np.ndarray:
 
    with ZipFile(zip_file_path) as zf:
        file_names = zf.namelist()[1:]
        file_names = sorted(
            file_names,
            key=lambda x: int(re.findall(r'[\d]+', x)[0]))

        data = zf.read(file_names[image_index])
        img = cv2.imdecode(np.frombuffer(data, np.uint8),
                            cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.resize(
                img, resize_shape, interpolation=cv2.INTER_NEAREST)
        return img

def resize_images(
    images: np.ndarray, resize_shape: Tuple[int, int]) -> np.ndarray:
  
    images = [cv2.resize(img, resize_shape, interpolation=cv2.INTER_NEAREST)
              for img in images]

    return np.array(images)

def get_train_val_test_dataset_split(
    images: np.ndarray, masks: np.ndarray, test_size: float = .15,
    val_size: float = .15, seed: int = 42
    ):
    
    X_train, X_test, y_train, y_test =  train_test_split(
        images, masks, test_size=test_size, shuffle=True, random_state=seed)
    X_train, X_val, y_train, y_val =  train_test_split(
        X_train, y_train, test_size=val_size, shuffle=True, random_state=seed)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
