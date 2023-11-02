import numpy as np
from PIL import Image
import torch as torch
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
import os
from einops import rearrange
from custom_transforms import *
import pydicom as dcm
from pydicom.pixel_data_handlers.util import apply_voi_lut
from monai.transforms import AsDiscrete


class breast_Dataset(Dataset):
    def __init__(self,path, train=True):
        self.path = path
        self.train = train
        self.train_path_list = []
        self.train_list = []

        self.label_path_list = []
        self.label_list = []

        self.train_path = path + "/input"
        self.label_path = path + "/target"

        
        for file in os.listdir(self.train_path):
            self.train_path_list.append(os.path.join(self.train_path,file))
        self.train_path_list.sort()
                
        for file in os.listdir(self.label_path):
            self.label_path_list.append(os.path.join(self.label_path,file))           
        self.label_path_list.sort()

    def __len__(self):
        return len(self.label_path_list)

    def __getitem__(self,idx):
        if self.train:
            self.transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Resize((512,512)),
                                                customRandomRotate(degrees=180,SEED=idx),
                                                customRandomHorizontalFlip(p=0.5,SEED=idx),
                                                #customRandomResizedCrop(SEED=idx,size=(256,256))
                                                ])
        else:
            self.transform = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Resize((512,512))
                                                 ])
            
            
        image_path = self.train_path_list[idx]

        slice = dcm.read_file(image_path)
        image = slice.pixel_array
        image = apply_voi_lut(image, slice)
        epsilon = 1e-10
        min_val = np.min(image)
        max_val = np.max(image)
        image = (image - min_val) / (max_val - min_val+epsilon)
        
        image = Image.fromarray(image)

        label_path = self.label_path_list[idx]
        label = np.array(Image.open(label_path).convert("L"))
        label = Image.fromarray(label)


        input_image = self.transform(image)
        target_image = self.transform(label)

        threshold = AsDiscrete(threshold=0.5)
        thresh = threshold(target_image)

        return input_image, thresh
    
if __name__ == "__main__":
    dataset_path = "/mount_folder/sampling/test/balance"
    dataset = tumor_Dataset(dataset_path)
    print(len(dataset))
    dataloader = DataLoader(dataset,batch_size=1,shuffle=True)

