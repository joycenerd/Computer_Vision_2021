from options import opt

from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import numpy as np
import torch

from pathlib import Path
import collections
import numbers
import random


label_dict = {
    'A': 0,
    'B': 1,
    'C': 2
}


class SceneDataset(Dataset):
    def __init__(self, root_dir, mode, transform=None):
        self.root_dir = Path(root_dir)
        self.x = []
        self.y = []
        self.transform = transform
        self.num_classes = opt.num_classes

        if mode == "train":
            labels = np.genfromtxt(Path(opt.data_root).joinpath(
                'train.csv'), dtype=np.str, delimiter=',', skip_header=1)

        else:
            labels = np.genfromtxt(Path(opt.data_root).joinpath(
                'eval.csv'), dtype=np.str, delimiter=',', skip_header=1)

        for label in labels:
            self.x.append(label[0])
            self.y.append(int(label[1]))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        image_path = self.x[index]
        image = Image.open(image_path).convert('RGB')
        image = image.copy()

        if self.transform:
            image = self.transform(image)

        return image, self.y[index]


def Dataloader(dataset, batch_size, shuffle, num_workers):
    data_loader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return data_loader


def _random_colour_space(x):
    output = x.convert("HSV")
    return output


class RandomShift(object):
    def __init__(self, shift):
        self.shift = shift

    @staticmethod
    def get_params(shift):
        """Get parameters for ``rotate`` for a random rotation.
        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        hshift, vshift = np.random.uniform(-shift, shift, size=2)

        return hshift, vshift

    def __call__(self, img):
        hshift, vshift = self.get_params(self.shift)

        return img.transform(img.size, Image.AFFINE, (1, 0, hshift, 0, 1, vshift), resample=Image.BICUBIC, fill=1)


def make_dataset(mode):

    colour_transform = transforms.Lambda(lambda x: _random_colour_space(x))

    transform = [
        transforms.RandomAffine(degrees=30, shear=50,
                                interpolation=InterpolationMode.BILINEAR, fill=0),
        transforms.RandomGrayscale(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomPerspective(
            distortion_scale=0.5, p=0.5, interpolation=InterpolationMode.BILINEAR, fill=0),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        transforms.Grayscale(num_output_channels=3),
        RandomShift(3),
        transforms.RandomApply([colour_transform]),
    ]

    data_transform_train = transforms.Compose([
        transforms.RandomResizedCrop(opt.img_size),
        transforms.RandomApply(transform, p=0.5),
        transforms.RandomApply([transforms.RandomRotation(
            (-90, 90), interpolation=InterpolationMode.BILINEAR, expand=False, center=None)], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    data_transform_dev = transforms.Compose([
        transforms.Resize((opt.img_size, opt.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    if(mode == "train"):
        data_set = SceneDataset(opt.data_root, mode, data_transform_train)
    elif(mode == "eval"):
        data_set = SceneDataset(opt.data_root, mode, data_transform_dev)

    return data_set
