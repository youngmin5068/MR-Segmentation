import torch as torch
import torchvision.transforms as transforms
from custom_transforms import *
import random 
import numpy as np

class customRandomHorizontalFlip(object):
    def __init__(self,SEED,p=0.5):
        self.p = p
        self.seed = SEED
    def __call__(self, image):
        # Set random seed
        seed = self.seed
        torch.manual_seed(seed)
        # Apply random transformation
        transform = transforms.RandomHorizontalFlip(p=self.p)
        image = transform(image)

        return image
    
class customRandomResizedCrop(object):
    def __init__(self, SEED, size, scale=(0.08, 1.0), ratio=(0.75, 1.333)):
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.seed = SEED

    def __call__(self, image):
        # Set random seed
        seed = self.seed
        torch.manual_seed(seed)
        # Apply random transformation
        transform = transforms.RandomResizedCrop(self.size, scale=self.scale, ratio=self.ratio)
        image = transform(image)

        return image

class customRandomVerticalFlip(object):
    def __init__(self, SEED=None, p=0.5):
        self.seed = SEED
        self.p = p
        
    def __call__(self, image):
        if self.seed is not None:
            torch.manual_seed(self.seed)
        if random.random() < self.p:
            return transforms.functional.vflip(image)
        return image

class customRandomRotate(object):
    def __init__(self, degrees,SEED):
        self.degrees = degrees
        self.seed = SEED
    def __call__(self, image):

        seed = self.seed
        torch.manual_seed(seed)
        # Generate random angle
        angle = transforms.RandomRotation.get_params((-self.degrees, self.degrees))
        # Apply random rotation
        image = transforms.functional.rotate(image, angle)

        return image
    
class customRandomCrop(object):
    def __init__(self, size, padding=None, seed=None):
        self.size = size
        self.padding = padding
        self.seed = seed

    def __call__(self, image):
        if self.padding is not None:
            image = transforms.functional.pad(image, self.padding)

        torch.manual_seed(self.seed)

        # Get image dimensions
        w, h = image.size

        # Get crop size
        th, tw = self.size
        if w == tw and h == th:
            return image

        # Get random crop position
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        # Apply crop
        return image.crop((x1, y1, x1 + tw, y1 + th))
    
