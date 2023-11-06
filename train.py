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


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)



def train_net(net,
              roi_model,          
              device,     
              epochs=EPOCHS,
              batch_size=BATCH_SIZE,
              lr=LEARNINGRATE,
              save_cp=True
              ):
    validation_interval = 20
    train_dataset = tumor_Dataset(path=TRAIN_PATH)
    tuning_dataset = tumor_Dataset(path=TUNING_PATH)
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

    optimizer = optim.AdamW(net.parameters(),betas=(0.9,0.999),lr=lr) # weight_decay : prevent overfitting
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=10,T_mult=1,eta_min=0.00001,last_epoch=-1)
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=[80],gamma=5)
    diceloss = DiceLoss()
    bceloss = nn.BCEWithLogitsLoss()
    threshold = AsDiscrete(threshold=0.5)
    best_epoch = 0
    best_dice = 0.0
    best_precision = 0.0
    best_recall = 0.0

    for epoch in range(epochs):

        net.train()
        roi_model.eval()
        i=1
        for imgs,true_masks in train_loader:

            imgs = imgs.to(device=device,dtype=torch.float32)
            true_masks = true_masks.to(device=device,dtype=torch.float32)

            for param in net.parameters():
                param.grad = None

            with torch.no_grad():
                roi_preds = torch.sigmoid(roi_model(imgs))
                roi_thresh = threshold(roi_preds)
                roi_results = imgs * roi_thresh

            masks_preds = net(roi_results)
            loss1 = diceloss(torch.sigmoid(masks_preds),true_masks)
            loss2 = bceloss(masks_preds,true_masks)
            

            loss = loss1+loss2
            loss.backward()

            nn.utils.clip_grad_value_(net.parameters(), 0.1)     

            optimizer.step()

            if i*batch_size%800 == 0:
                print('epoch : {}, index : {}/{}, dice loss : {:.4f}, bce loss : {:.4f}, total loss : {:.4f}'.format(
                                                                                epoch+1, 
                                                                                i*batch_size,
                                                                                train_size,
                                                                                loss1.detach(),
                                                                                loss2.detach(),
                                                                                loss.detach())) 
            i += 1
        
        #when train epoch end
        print("--------------Validation start----------------")
        net.eval()      
        dice = 0.0
        recall = 0.0
        precision = 0.0

        TP_total = 0 
        FP_total = 0 
        FN_total = 0 
        TN_total = 0 

        for imgs, true_masks in val_loader:
            imgs = imgs.to(device=device,dtype=torch.float32)
            true_masks = true_masks.to(device=device,dtype=torch.float32)

            with torch.no_grad():
                roi_preds = torch.sigmoid(roi_model(imgs))
                roi_thresh = threshold(roi_preds)
                roi_results = imgs * roi_thresh

                mask_pred = net(roi_results)
                mask_pred = torch.sigmoid(mask_pred)

            pred_thresh = threshold(mask_pred)

            predicted_mask = pred_thresh.cpu().numpy()
            true_mask = true_masks.cpu().numpy()

            # 이진 분류를 위한 threshold 적용
            predicted_mask = (predicted_mask > 0.5).astype(np.uint8)
            
            # 현재 slice에 대한 confusion matrix 요소 계산
            TP = np.sum((predicted_mask == 1) & (true_mask == 1))
            FP = np.sum((predicted_mask == 1) & (true_mask == 0))
            FN = np.sum((predicted_mask == 0) & (true_mask == 1))
            TN = np.sum((predicted_mask == 0) & (true_mask == 0))
            
            # 현재 slice의 결과를 전체 결과에 합산
            TP_total += TP
            FP_total += FP
            FN_total += FN
            TN_total += TN
    
            # 모든 slices를 기반으로 한 Dice Score 계산
            dice = (2 * TP_total) / (2 * TP_total + FP_total + FN_total)

            # precision += precision_score(pred_thresh,true_masks)
            # recall += recall_score(pred_thresh,true_masks)
            #dice += dice_score(pred_thresh,true_masks)
            
        print("dice score : {:.4f}".format(dice))
        # print("dice score : {:.4f}, len(val_loader) : {:.4f}".format(dice, len(val_loader)))
        # print("dice score : {:.4f}, recall score : {:.4f}, precision score : {:.4f}".format(dice/len(val_loader), recall/len(val_loader),precision/len(val_loader)) )
        scheduler.step()
        
        if dice > best_dice:
            best_dice = dice
            # best_recall = recall/len(val_loader)
            # best_precision = precision/len(val_loader)
            best_epoch = epoch+1

            if save_cp:
                try:
                    os.mkdir(DIR_CHECKPOINT)
                    logging.info("Created checkpoint directory")
                except OSError:
                    pass
                torch.save(net.state_dict(), DIR_CHECKPOINT + f'/Deform_UNet_231105.pth')
                logging.info(f'Checkpoint {epoch + 1} saved !')
        print("best_dice : {:.4f}".format(best_dice))
        #print("epoch : {} , best_dice : {:.4f}, best_recall : {:.4f}, best_precision : {:.4f}".format(best_epoch, best_dice,best_recall,best_precision))

if __name__ == '__main__':
    Model_SEED = 7777777
    set_seed(Model_SEED)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    #net = tumor_model(img_size=(512,512),spatial_dims=2,in_channels=1,out_channels=1,depths=(2,2,2,2)).to(device=device)
    net = Deform_UNet(1,1).to(device=device)
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net,device_ids=[0,1,2,3])


    model_path = ROI_MODEL_PATH
    roi_model = ROI_MODEL(img_size=(512,512),spatial_dims=2,in_channels=1,out_channels=1,depths=(2,2,2,2)).to(device=device)
    if torch.cuda.device_count() > 1:
        roi_model = nn.DataParallel(roi_model,device_ids=[0,1,2,3]) 

    roi_model.load_state_dict(torch.load(model_path))

    train_net(net=net,roi_model=roi_model,batch_size=BATCH_SIZE,lr=LEARNINGRATE,epochs=EPOCHS,device=device)



