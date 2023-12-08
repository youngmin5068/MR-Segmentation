import numpy as np
from PIL import Image
import torch as torch
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
import os
import cv2
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
        self.label_path_list = []
        self.mask_path_list = []
 

        self.train_path = path + "/input"
        self.label_path = path + "/target"
        self.mask_path = path + "/breast_roi"

        
        for file in os.listdir(self.train_path):
            self.train_path_list.append(os.path.join(self.train_path,file))
        self.train_path_list.sort()
                
        for file in os.listdir(self.label_path):
            self.label_path_list.append(os.path.join(self.label_path,file))           
        self.label_path_list.sort()

        for file in os.listdir(self.mask_path):
            self.mask_path_list.append(os.path.join(self.mask_path,file))           
        self.mask_path_list.sort()


    def __len__(self):
        return len(self.label_path_list)

    def preprocessing(self,train_path, label_path, mask_path):
        
        input_slice = pydicom.read_file(train_path)
        input_img = input_slice.pixel_array
        #input_img = apply_voi_lut(input_img, input_slice)
        epsilon = 1e-10
        min_val = np.min(input_img)
        max_val = np.max(input_img)
        input_img = (input_img - min_val) / (max_val - min_val+epsilon)
        

        target_slice = pydicom.read_file(label_path)
        target_img = target_slice.pixel_array
        epsilon = 1e-10
        min_val = np.min(target_img)
        max_val = np.max(target_img)
        target_img = (target_img - min_val) / (max_val - min_val+epsilon)
        

        np_img = np.array(Image.open(mask_path))
        np_img = np_img/255.0


        contours, _ = cv2.findContours(np_img.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        rects = []

        for contour in contours:
            rects.append(cv2.boundingRect(contour))

        if len(rects) > 0:
            y_min = min(rect[1] for rect in rects)
            y_max = max(rect[1] + rect[3] for rect in rects)

            input_img = input_img[y_min:y_max,:]
            target_img = target_img[y_min:y_max,:]
        else:
            input_img = input_img[200:320,:]
            target_img = target_img[200:320,:]

        input_img = Image.fromarray(input_img)
        target_img = Image.fromarray(target_img)


        return input_img, target_img



    def __getitem__(self,idx):
        if self.train:
            self.transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Resize((256,512)),
                                                customRandomRotate(degrees=10,SEED=idx),
                                                customRandomHorizontalFlip(p=0.5,SEED=idx),
                                                customRandomVerticalFlip(p=0.5,SEED=idx)
                                                #customRandomResizedCrop(SEED=idx, size=(256,256))
                                                ])
        else:
            self.transform = transforms.Compose([transforms.ToTensor(),
                                                 #transforms.Resize((256,256)),
                                                 #customRandomResizedCrop(SEED=idx, size=(256,256))
                                                 ])


        image,label = self.preprocessing(self.train_path_list[idx], self.label_path_list[idx],self.mask_path_list[idx])    

        contrast = transforms.ColorJitter(contrast=(0,1.5))

        input_image = self.transform(image)
        target_image = self.transform(label)

        threshold = AsDiscrete(threshold=0.5)
        target_image = threshold(target_image)


        return input_image, target_image
    
    
if __name__ == "__main__":
    dataset_path = "/mount_folder/Tumors/train/undersampling"
    dataset = tumor_Dataset(dataset_path)
    dataloader = DataLoader(dataset,batch_size=1,shuffle=True)

    print(next(iter(dataloader))[1].shape)

