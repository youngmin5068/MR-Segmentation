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
import cv2

class tumor_Dataset(Dataset):
    def __init__(self,path, train=True):
        self.path = path
        self.train = train

        patients = [f for f in os.listdir(self.path)]
        patients.sort()

        self.train_path_list = []
        self.label_path_list = []
        self.mask_path_list = []

        if self.train:
            weird =['5140749',
                    '7280265',
                    '1649257',
                    '7871065',
                    '3033890',
                    '3149487',
                    '3559627',
                    '4716956',
                    '3849047',
                    '6968225']
            patients = [f for f in patients if str(f) not in weird]

        for p in patients:
            train_path = [self.path+f"{p}/undersampling/input/" + f for f in  os.listdir(self.path+f"{p}/undersampling/input")]
            train_path.sort()
            self.train_path_list = self.train_path_list + train_path

            label_path = [self.path+f"{p}/undersampling/target/" + f for f in  os.listdir(self.path+f"{p}/undersampling/target")]
            label_path.sort()
            self.label_path_list = self.label_path_list + label_path

            mask_path = [self.path+f"{p}/undersampling/breast_roi/" + f for f in os.listdir(self.path+f"{p}/undersampling/breast_roi")]
            mask_path.sort()
            self.mask_path_list = self.mask_path_list + mask_path

        self.train_path_list.sort()
        self.label_path_list.sort()
        self.mask_path_list.sort()
 

    
    def __len__(self):
        return len(self.mask_path_list)

    def preprocessing(self,train_path, label_path, mask_path):
        
        input_slice = pydicom.read_file(train_path)
        positions = pydicom.dcmread(train_path).ImagePositionPatient

        input_img = input_slice.pixel_array
        input_img = apply_voi_lut(input_img, input_slice)
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


        if positions[1] < 0 :
            input_img = np.flipud(input_img)
            target_img = np.flipud(target_img)
            np_img = np.flipud(np_img)


        contours, _ = cv2.findContours(np_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        rects = []

        for contour in contours:
            rects.append(cv2.boundingRect(contour))

        
        y_min = min(rect[1] for rect in rects)
        y_max = max(rect[1] + rect[3] for rect in rects)

        y_length = y_max - y_min


        if y_length < 192:
            crop_input_img = input_img[y_min-(192 - y_length)//2:y_max + (192-y_length)//2,:]
            crop_target_img = target_img[y_min-(192 - y_length)//2:y_max + (192-y_length)//2,:]
    
        else:
            crop_input_img = input_img[y_min:y_max,:]
            crop_target_img = target_img[y_min:y_max,:]




        input_img = Image.fromarray(input_img)
        target_img = Image.fromarray(target_img)



        return input_img, target_img,crop_input_img,crop_target_img



    def __getitem__(self,idx):
        if self.train:
            self.transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Resize((512,512)),
                                                customRandomRotate(degrees=10,SEED=idx),
                                                customRandomHorizontalFlip(p=0.5,SEED=idx),
                                                customRandomVerticalFlip(p=0.5,SEED=idx)
                                                #customRandomResizedCrop(SEED=idx, size=(256,256))
                                                ])
            self.crop_transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Resize((192,512)),
                                                customRandomRotate(degrees=10,SEED=idx),
                                                customRandomHorizontalFlip(p=0.5,SEED=idx),
                                                customRandomVerticalFlip(p=0.5,SEED=idx)
                                                #customRandomResizedCrop(SEED=idx, size=(256,256))
                                                ])
        else:
            self.transform = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Resize((512,512)),
                                                 #customRandomResizedCrop(SEED=idx, size=(256,256))
                                                 ])
            self.crop_transform = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Resize((512,512)),
                                                 #customRandomResizedCrop(SEED=idx, size=(256,256))
                                                 ])


        image,label,crop_img,crop_label = self.preprocessing(self.train_path_list[idx], self.label_path_list[idx],self.mask_path_list[idx])    
        
        input_image = (self.transform(image))
        target_image = (self.transform(label))

        crop_img = self.crop_transform(crop_img)
        crop_label = self.crop_transform(crop_label)

        threshold = AsDiscrete(threshold=0.5)
        target_image = threshold(target_image)
        crop_label = threshold(crop_label)



        return input_image, target_image, crop_img, crop_label
    

# dataset = tumor_Dataset(path="/mount_folder/New_Tumors/tuning/")

# loader = DataLoader(dataset,batch_size=1,shuffle=True)

# sample = next(iter(loader))

# print(torch.unique(sample[1]))