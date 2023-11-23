import numpy as np
from PIL import Image
import torch as torch
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
import os
from einops import rearrange
from custom_transforms import *
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from monai.transforms import AsDiscrete


class tumor_Dataset(Dataset):
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

    def preprocessing(self,train_path, label_path):

        input_slice = pydicom.read_file(train_path)
        input_img = input_slice.pixel_array
        input_img = apply_voi_lut(input_img, input_slice)
        epsilon = 1e-10
        min_val = np.min(input_img)
        max_val = np.max(input_img)
        input_img = (input_img - min_val) / (max_val - min_val+epsilon)
        input_img = Image.fromarray(input_img)

        target_slice = pydicom.read_file(label_path)
        target_img = target_slice.pixel_array
        epsilon = 1e-10
        min_val = np.min(target_img)
        max_val = np.max(target_img)
        target_img = (target_img - min_val) / (max_val - min_val+epsilon)
        target_img = Image.fromarray(target_img)

        return input_img, target_img

    def resize(self,size):
        
        return  transforms.Resize((size,size))

    def __getitem__(self,idx):
        if self.train:
            self.transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Resize((512,512)),
                                                customRandomRotate(degrees=180,SEED=idx),
                                                customRandomHorizontalFlip(p=0.5,SEED=idx),
                                                #customRandomResizedCrop(SEED=idx, size=(256,256))
                                                ])
        else:
            self.transform = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Resize((512,512)),
                                                 #customRandomResizedCrop(SEED=idx, size=(256,256))
                                                 ])


        image,label = self.preprocessing(self.train_path_list[idx], self.label_path_list[idx])    

        input_image = self.transform((image))
        target_image = self.transform((label))

        threshold = AsDiscrete(threshold=0.5)
        target_image = threshold(target_image)

        target_1 = self.resize(256)(target_image)
        target_2 = self.resize(128)(target_image)
        target_3 = self.resize(64)(target_image)
        target_4 = self.resize(32)(target_image)


        return input_image, [target_image,target_1,target_2,target_3,target_4]
    
if __name__ == "__main__":
    dataset_path = "/mount_folder/Tumors/train/undersampling"
    dataset = tumor_Dataset(dataset_path)
    dataloader = DataLoader(dataset,batch_size=1,shuffle=True)

    print(next(iter(dataloader))[1][1].shape)

