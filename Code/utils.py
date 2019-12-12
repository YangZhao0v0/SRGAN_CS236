import numpy as np
import os
import shutil
import sys
import torch
from PIL import Image
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize


bce = torch.nn.BCEWithLogitsLoss(reduction='none')

def train_hr_transform(crop_size):
    return Compose([
        RandomCrop(crop_size,pad_if_needed=True),
        ToTensor(),
    ])


def train_lr_transform(crop_size, upscale_factor):
    return Compose([
        ToPILImage(),
        Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
        ToTensor()
    ])

def val_hr_transform(crop_size):
    return Compose([
        ToPILImage(),
        Resize(crop_size, interpolation=Image.BICUBIC),
        ToTensor()
    ])

class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super().__init__()
        self.image_filenames = [os.path.join(dataset_dir, x) for x in os.listdir(dataset_dir) if x.endswith('jpg')]
        self.hr_transform = train_hr_transform(crop_size)
        self.lr_transform = train_lr_transform(crop_size, upscale_factor)

    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.image_filenames[index]))
        lr_image = self.lr_transform(hr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)

class ValDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.image_filenames = [os.path.join(dataset_dir, x) for x in os.listdir(dataset_dir) if x.endswith('jpg')]
        self.hr_transform = train_hr_transform(crop_size)
        self.lr_transform = train_lr_transform(crop_size, upscale_factor)
        self.restore_transform = val_hr_transform(crop_size)

    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.image_filenames[index]))
        lr_image = self.lr_transform(hr_image)
        hr_restore = self.restore_transform(lr_image)
        return lr_image, hr_image, hr_restore

    def __len__(self):
        return len(self.image_filenames)


# Visualization module:
def Reconstruct(lr_image, hr_image, hr_fake_image, dir):
    toImage = transforms.ToPILImage()
    figure, (lr, hr, hr_fake) = plt.subplots(1,3)
    lr.imshow(toImage(lr_image))
    lr.axis('off')
    hr.imshow(toImage(hr_image))
    hr.axis('off')
    hr_fake.imshow(toImage(hr_fake_image.clamp(0,1)))
    hr_fake.axis('off')
    plt.savefig(dir, bbox_inches='tight')

# Save learning curve:
def LearningCurve(arr, dir, isG):
    plt.close('all')
    if isG:
        fig, ax1 = plt.subplots()
        color = 'tab:red'
        ax1.set_xlabel('# of batches')
        ax1.set_ylabel('total generator loss', color=color)
        ax1.plot(arr[0], color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:blue'
        ax2.set_ylabel('adversarial generator loss', color=color)  # we already handled the x-label with ax1
        ax2.plot(arr[2], color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
    else:
        plt.xlabel('# of batches')
        plt.ylabel('Loss')
        plt.plot(arr[0])
        plt.legend(['Discriminator Loss'])
    plt.savefig(dir, bbox_inches='tight')

# System verifier
if sys.version_info[0] < 3:
    raise Exception("Detected unpermitted Python version: Python{}. You should use Python3."
                    .format(sys.version_info[0]))
