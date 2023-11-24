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
from Small_dataset import tumor_Dataset
from dataloader import data_load
from Nested_UNet import NestedUNet
from custom_UNet_v2 import custom_UNet_V2_small
from custom_UNet_v1 import custom_UNet_V1
from tumorSeg_model import tumor_model
from UNet import UNet

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)



def train_net(net,      
              device,     
              epochs=EPOCHS,
              batch_size=BATCH_SIZE,
              lr=LEARNINGRATE,
              save_cp=True
              ):

    train_dataset = tumor_Dataset(path="/mount_folder/Tumors/small_train")
    tuning_dataset = tumor_Dataset(path="/mount_folder/Tumors/small_tuning")
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
    #scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=10,T_mult=2,eta_min=0.00005,last_epoch=-1)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=[100],gamma=5)
    diceloss = DiceLoss()
    bceloss = nn.BCEWithLogitsLoss()
   
    threshold = AsDiscrete(threshold=0.5)

    best_epoch = 0 
    best_dice = 0.0
    patience_limit = 200 # 몇 번의 epoch까지 지켜볼지를 결정
    patience_check = 0

    for epoch in range(epochs):

        net.train()
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

            masks_preds = net(imgs)
            loss1 = diceloss(torch.sigmoid(masks_preds),true_masks)
            loss2 = bceloss(masks_preds, true_masks)


            #loss2 = bceloss(masks_preds,true_masks)
            loss = loss1 + loss2
            loss.backward()

            nn.utils.clip_grad_value_(net.parameters(), 0.1)     

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

        val_loss = 0.0
        dice = 0.0
        for imgs, true_masks in val_loader:
            imgs = imgs.to(device=device,dtype=torch.float32)
            true_masks = true_masks.to(device=device,dtype=torch.float32)

            with torch.no_grad():
                mask_pred = net(imgs)

            pred_thresh = threshold(mask_pred)
            
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
                torch.save(net.state_dict(), DIR_CHECKPOINT + f'/UNet_smallTumor_v3.pth') #v1 - Nested UNet v2 - swinunetr v3 - custom_UNetv1 수정본
                logging.info(f'Checkpoint {epoch + 1} saved !')

        if patience_check >= patience_limit: # early stopping 조건 만족 시 조기 종료
            break

        print("best epoch : {}, best dice : {:.4f}".format(best_epoch+1,best_dice))
               
  
        scheduler.step()
        #print("epoch : {} , best_dice : {:.4f}, best_recall : {:.4f}, best_precision : {:.4f}".format(best_epoch, best_dice,best_recall,best_precision))

if __name__ == '__main__':
    Model_SEED = 7777777
    set_seed(Model_SEED)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    #net = tumor_model(img_size=(512,512),spatial_dims=2,in_channels=1,out_channels=1,depths=(2,2,2,2),feature_size=36).to(device=device)
    #net = NestedUNet(1,1).to(device=device)
    net = custom_UNet_V1(1,1).to(device=device)
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net,device_ids=[0,1,2,3])


    # model_path = B_DIR_CHECKPOINT + '/ROI_Model_231114.pth'
    # roi_model = tumor_model(img_size=(512,512),spatial_dims=2,in_channels=1,out_channels=1,depths=(2,2,2,2),feature_size=24).to(device=device)
    # if torch.cuda.device_count() > 1:
    #     roi_model = nn.DataParallel(roi_model,device_ids=[0,1,2,3]) 

    # roi_model.load_state_dict(torch.load(model_path))

    train_net(net=net,batch_size=16,lr=0.0001,epochs=500,device=device)




