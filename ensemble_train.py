import os
import numpy as np
import random
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from config import *
from loss import *
from metric import *
from monai.transforms import AsDiscrete
from dataset import tumor_Dataset
from dataloader import data_load
from tumorSeg_model import tumor_model
from Deform_LKA import Deform_UNet
from roi_model import ROI_MODEL
from ACC_UNet import ACC_UNet_Lite
from Nested_UNet import NestedUNet
import torchvision.transforms as transforms
from custom_UNet_v2 import custom_UNet_V2
from custom_UNet_v1 import custom_UNet_V1
from custom_UNet_v3 import custom_UNet_V3


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)



def train_net(net,
              net2,      
              device,     
              epochs=EPOCHS,
              batch_size=BATCH_SIZE,
              lr=LEARNINGRATE,
              save_cp=True
              ):

    train_dataset = tumor_Dataset(path="/mount_folder/Tumors/train/undersampling")
    tuning_dataset = tumor_Dataset(path="/mount_folder/Tumors/tuning/undersampling")
    train_loader, train_size = data_load(train_dataset,batch_size=batch_size,train=True,shuffle=True)
    val_loader,val_size = data_load(tuning_dataset,batch_size=batch_size,train=False,shuffle=False)

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Train size:      {train_size}
        Test size:       {val_size}
        Learning rate:   {lr}        
        Checkpoints:     {save_cp}
        Device:          {device}
    ''')    

    optimizer = optim.AdamW(net.parameters(),betas=(0.9,0.999),lr=lr,weight_decay=1e-5) # weight_decay : prevent overfitting
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=10,T_mult=1,eta_min=0.0001,last_epoch=-1)
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=[50],gamma=5)
    diceloss = DiceLoss()
    bceloss = nn.BCEWithLogitsLoss()
   
    threshold = AsDiscrete(threshold=0.5)

    best_epoch = 0 
    best_dice = 0.0
    patience_limit = 500 # 몇 번의 epoch까지 지켜볼지를 결정
    patience_check = 0

    for epoch in range(epochs):

        net.train()
        net2.train()
        #roi_model.eval()
        i=1
        for imgs,true_masks in train_loader:
    
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device,dtype=torch.float32)
            for param in net.parameters():
                param.grad = None

            # with torch.no_grad():
            #     roi_preds = torch.sigmoid(roi_model(imgs))
            #     roi_thresh = threshold(roi_preds)
            #     roi_results = imgs * roi_thresh

            mask_pred1 = net(imgs)
            mask_pred2 = net2(imgs)

            mask_pred2 = mask_pred2.to(mask_pred1.device)

            loss1 = diceloss(torch.sigmoid((mask_pred1+mask_pred2)/2),true_masks)
            loss2 = bceloss((mask_pred1+mask_pred2)/2, true_masks)
   

            #loss2 = bceloss(masks_preds,true_masks)ㅊ
            loss = loss1 + loss2 
            loss.backward()

            #nn.utils.clip_grad_value_(net.parameters(), 0.1)  

            optimizer.step()

            if i*batch_size%800 == 0:
                print('epoch : {}, index : {}/{}, total loss: {:.4f}'.format(
                                                                                epoch+1, 
                                                                                i*batch_size,
                                                                                train_size,
                                                                                loss.detach())
                                                                                ) 
            i += 1
        
        #when train epoch end
        print("--------------Validation start----------------")
        net.eval()      
        net2.eval()

        val_loss = 0.0
        dice = 0.0
        for imgs, true_masks in val_loader:
            imgs = imgs.to(device=device,dtype=torch.float32)
            true_masks = true_masks.to(device=device,dtype=torch.float32)

            with torch.no_grad():
                mask_pred1 = net(imgs)
                mask_pred2 = net2(imgs)

                mask_pred2 = mask_pred2.to(mask_pred1.device)

            pred_thresh = threshold(torch.sigmoid((mask_pred1+mask_pred2)/2))
            
            dice += dice_score(pred_thresh, true_masks)

        mean_dice_score = dice/len(val_loader)

        if mean_dice_score < best_dice: # dice가 개선되지 않은 경우
            print("current dice : {:.4f}".format(mean_dice_score))
            patience_check += 1
        else:
            print("UPDATE dice, loss")
            best_epoch = epoch
            best_dice = mean_dice_score
            patience_check = 0 
            if save_cp:
                try:
                    os.mkdir(DIR_CHECKPOINT)
                    logging.info("Created checkpoint directory")
                except OSError:
                    pass
                torch.save(net.state_dict(), DIR_CHECKPOINT + f'/ensemble_model1_exclude_tumor_model_with_customV1.pth') # d-> custom_UNet_V2 /// e -> att swinUNetr  ////f -> custom unet v2 //ensemble
                torch.save(net2.state_dict(), DIR_CHECKPOINT + f'/ensemble_model2_exclude_customV1.pth') #ensemble_model2 - custom v2, ensemble_model2_b - custom v1
                logging.info(f'Checkpoint {epoch + 1} saved !')

        if patience_check >= patience_limit: # early stopping 조건 만족 시 조기 종료
            break

        print("best epoch : {}, best dice : {:.4f}".format(best_epoch+1,best_dice))
               
  
        scheduler.step()
    

if __name__ == '__main__':
    Model_SEED = 7777777
    set_seed(Model_SEED)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = tumor_model(img_size=(512,512),spatial_dims=2,in_channels=1,out_channels=1,depths=(2,2,2,2),feature_size=24).to(device=device)
    net2 = custom_UNet_V1(1, 1).to(device=device)
    #net = custom_UNet_V2(1,1).to(device=device)
    #net = ACC_UNet_Lite(1,1).to(device=device)
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net,device_ids=[0,1,2,3,4,5])
        net2 = nn.DataParallel(net2,device_ids=[0,1,2,3,4,5])


    # model_path = B_DIR_CHECKPOINT + '/ROI_Model_231114.pth'
    # roi_model = tumor_model(img_size=(512,512),spatial_dims=2,in_channels=1,out_channels=1,depths=(2,2,2,2),feature_size=24).to(device=device)
    # if torch.cuda.device_count() > 1:
    #     roi_model = nn.DataParallel(roi_model,device_ids=[0,1,2,3]) 

    # roi_model.load_state_dict(torch.load(model_path))

    train_net(net,net2,batch_size=32,lr=0.001,epochs=EPOCHS,device=device)




