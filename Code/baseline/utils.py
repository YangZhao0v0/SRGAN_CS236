import numpy as np
import os
import shutil
import sys
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

bce = torch.nn.BCEWithLogitsLoss(reduction='none')


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
    plt.xlabel('# of batches')
    plt.ylabel('Loss')
    if isG:
        plt.plot(arr[0])
        plt.plot(arr[1])
        plt.plot(arr[2])
        plt.legend(['Generator Loss', 'Generator Loss -- content', 'Generator Loss -- adversarial'])

    else:
        plt.plot(arr[0])
        plt.legend(['Discriminator Loss'])
    plt.savefig(dir, bbox_inches='tight')

# System verifier
if sys.version_info[0] < 3:
    raise Exception("Detected unpermitted Python version: Python{}. You should use Python3."
                    .format(sys.version_info[0]))
