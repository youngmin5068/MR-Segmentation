import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets):

        intersection = torch.sum(inputs * targets)
        union = torch.sum(inputs) + torch.sum(targets)
        dice = (2 * intersection) / (union + 1e-8)  # Adding a small epsilon to avoid division by zero
        dice_loss = 1 - dice
        return dice_loss

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.6, beta=0.4):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, inputs, targets):
        TP = torch.sum(inputs * targets)
        FP = torch.sum(inputs * (1 - targets))
        FN = torch.sum((1 - inputs) * targets)

        tversky_index = (TP) / (TP + self.alpha * FN + self.beta * FP + 1e-8)
        tversky_loss = 1 - tversky_index

        return tversky_loss


class marginBCELoss(nn.Module):
    def __init__(self, s=30, m=0.35):
        super(marginBCELoss, self).__init__()
        self.s = s
        self.m = m
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, logits, labels):
        # logits는 모델의 출력값이며, cos(θ) 값을 가정합니다.
        # labels는 실제 레이블이며, [0, 1]의 값을 가정합니다.
        
        # I(y)를 계산합니다.
        Iy = labels.float()
        
        # cos(θ) - I(y)m 값을 계산합니다.
        adjusted_logits = logits - Iy * self.m
        
        # s * (cos(θ) - I(y)m) 값을 계산합니다.
        scaled_logits = self.s * adjusted_logits
        
        # 손실값을 계산합니다.
        loss = self.criterion(scaled_logits, labels)
        
        return loss

