import torch
from dataloader import data_load
from metric import *
import torch.nn as nn
from dataset import tumor_Dataset
from config import *
from torch.utils.data import DataLoader
from monai.networks.nets import  SwinUNETR
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model',required=True,help="require trained model (.pth)")
args = parser.parse_args()

test_loader,test_size = data_load(path=TEST_PATH,batch_size=BATCH_SIZE,train=False,shuffle=False)
print("test_size: ", test_size)

device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')

model_path = DIR_CHECKPOINT + f"/{args.model}"

net = SwinUNETR(img_size=(512,512),spatial_dims=2,in_channels=1,out_channels=1,depths=(2,2,2,2)).to(device=device)
if torch.cuda.device_count() > 1:
    net = nn.DataParallel(net,device_ids=[0,1,2,3])

net.load_state_dict(torch.load(model_path))
    

print("--------------------------TEST------------------------")

net.eval()
precision=0.0
recall=0.0
dice=0.0

for imgs, true_masks in test_loader:
    imgs = imgs.to(device=device,dtype=torch.float32)
    true_masks = true_masks.to(device=device,dtype=torch.float32)

    with torch.no_grad():
        mask_pred = net(imgs)
        mask_pred = torch.sigmoid(mask_pred)
        thresh = torch.zeros_like(mask_pred)
        thresh[mask_pred > 0.5] = 1.0

    precision += precision_score(thresh,true_masks)
    recall += recall_score(thresh,true_masks)
    dice += dice_score(thresh,true_masks)

print("dice score : {:.4f}, len(testloader) : {:.4f}".format(dice, len(test_loader)))
print("dice score : {:.4f}, recall score : {:.4f}, precision score : {:.4f}".format(dice/len(test_loader), recall/len(test_loader),precision/len(test_loader)) )