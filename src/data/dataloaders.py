from typing import Tuple
import torch
from torch.utils.data.dataloader import Dataset, DataLoader
import torchvision.transforms as T
import numpy as np


class Dataset(Dataset):
    def __init__(self, X, y, train_mean,
                 train_std, apply_augmentation):

        super().__init__()
        assert X.shape[:-1] == y.shape, 'The image does not have the same ' +\
            'shape as the mask.'
        self.X = X
        self.y = y
        self.len = X.shape[0]
        self.transform_image_and_mask = T.Compose([
            T.RandomPerspective(distortion_scale=.3),
            T.RandomHorizontalFlip(),
            T.RandomAffine(degrees=(-45, 45), translate=(0.1, 0.1),
                           scale=(0.5, 1.5))
            ]) if apply_augmentation else None
        self.transform_image = T.ColorJitter(brightness=.2, hue=.05) \
            if apply_augmentation else None
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(mean=train_mean, std=train_std)

    def __getitem__(self, index ) :
        X, y = self.X[index], self.y[index]
        X = self.to_tensor(X)
        y[y == 1] = 255
        y = self.to_tensor(y)

        if self.transform_image_and_mask is not None:
            y = y.expand_as(X)
            X_y = torch.cat([X.unsqueeze(0), y.unsqueeze(0)], dim=0)
            X_y = self.transform_image_and_mask(X_y)
            X, y = X_y[0], X_y[1]
            y = y[:1]

        if self.transform_image is not None:
            X = self.transform_image(X)
        X = self.normalize(X)

        y_b = (~y.bool()).float()
        y = y.bool().float()
        y = torch.cat([y_b, y], dim=0)
        return X, y

    def __len__(self) -> int:
        return self.len

def get_dataloader(X, y, train_mean, train_std, batch_size, shuffle,apply_augmentation):
    return DataLoader(
        Dataset(X, y, train_mean, train_std,
                             apply_augmentation=apply_augmentation),
        batch_size=batch_size, shuffle=shuffle, drop_last=True)


