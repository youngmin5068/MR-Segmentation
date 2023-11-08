import torch

def recall_score(y_true, y_pred, eps=1e-7):
    y_true = y_true.float()
    y_pred = y_pred.float()
    tp = (y_true * y_pred).sum()
    fn = (y_true * (1 - y_pred)).sum()
    recall = tp / (tp + fn + eps)
    return recall

def precision_score(y_true, y_pred, eps=1e-7):
    y_true = y_true.float()
    y_pred = y_pred.float()
    tp = (y_true * y_pred).sum()
    fp = ((1 - y_true) * y_pred).sum()
    precision = tp / (tp + fp + eps)
    return precision

def dice_score(y_true, y_pred, eps=1e-9):

    y_true = y_true.float()
    y_pred = y_pred.float()
    
    intersection = torch.sum(y_true * y_pred)
    union = torch.sum(y_true) + torch.sum(y_pred)
    
    dice = (2. * intersection + eps) / (union + eps)
    return dice

