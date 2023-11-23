import torch
import torch.nn as nn
from Blocks import ChannelGate,custom_LKA,Topt_CBAM

class DoubleConv(nn.Module):
    def __init__(self,in_channels, out_channels):
        super().__init__()
        self.conv7x7 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding = 1, groups = in_channels)
        self.conv1x1_1 = nn.Conv2d(in_channels, 2*in_channels, kernel_size=1)
        self.norm = nn.GroupNorm(2*in_channels,2*in_channels)
        
        self.conv3x3 = nn.Conv2d(2*in_channels, 2*in_channels, kernel_size=3, padding=1, groups=2*in_channels)
        self.conv1x1_2 = nn.Conv2d(2*in_channels,out_channels,kernel_size=1)
        self.norm2 = nn.GroupNorm(out_channels,out_channels)
        
        self.act = nn.GELU()

    def forward(self,x):
        x = self.conv7x7(x)
        x = self.conv1x1_1(x)
        x = self.norm(x)

        x = self.conv3x3(x)
        x = self.conv1x1_2(x)
        x = self.act(x)

        return x
    
class Bottom(nn.Module):
    def __init__(self,in_channels, out_channels):
        super().__init__()
        self.conv7x7 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding = 1,groups = in_channels)
        self.conv1x1_1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.norm = nn.GroupNorm(in_channels,in_channels)
        
        self.conv3x3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.conv1x1_2 = nn.Conv2d(in_channels,out_channels,kernel_size=1)
        self.norm2 = nn.GroupNorm(out_channels,out_channels)
        
        self.act = nn.GELU()

    def forward(self,x):
        x = self.conv7x7(x)
        x = self.conv1x1_1(x)
        x = self.norm(x)

        x = self.conv3x3(x)
        x = self.conv1x1_2(x)
        x = self.act(x)

        return x


class Down(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.s2d = nn.PixelUnshuffle(downscale_factor=2)
        self.conv = nn.Conv2d(dim,dim,3,padding=1,groups=dim)
        self.conv1x1 = nn.Conv2d(dim,dim,1)
        self.norm = nn.GroupNorm(dim,dim)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.s2d(x)
        u = x.clone()
        x = self.conv(x)
        x = self.conv1x1(x)
        x = self.norm(x)
        x = self.act(x)


        return x+u

class Up(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.d2s = nn.PixelShuffle(upscale_factor=2)
        self.conv = nn.Conv2d(dim,dim,3,padding=1,groups=dim)
        self.conv1x1 = nn.Conv2d(dim,dim,1)
        self.norm = nn.GroupNorm(dim,dim)
        self.act = nn.GELU()

    
    def forward(self,x):
        x = self.d2s(x)
        u = x.clone()
        x = self.conv(x)
        x = self.conv1x1(x)
        x = self.norm(x)
        x = self.act(x)

        return x+u
    
class attentionGate(nn.Module):
    def __init__(self,dim,percent_t):
        super().__init__()
        self.channelGate = Topt_CBAM(dim,percent_t)
        self.lka = custom_LKA(dim)
    def forward(self, x):
        u = x.clone()
        x = self.channelGate(x)
        x = self.lka(x) + u
        return x


class ResPath(torch.nn.Module):
    """
    Implements ResPath-like modified skip connection

    """

    def __init__(self, in_chnls, n_lvl,percent_t):
        """
        Initialization

        Args:
            in_chnls (int): number of input channels
            n_lvl (int): number of blocks or levels
        """

        super(ResPath, self).__init__()

        self.n_lvl = n_lvl
        self.conv = attentionGate(in_chnls,percent_t)
    def forward(self, x):

        for _ in range(self.n_lvl):
            x = self.conv(x) + x

        return x



class Outc(nn.Module):
    def __init__(self, in_channels,num_classes):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,num_classes,1)
    def forward(self,x):
        return self.conv(x)


class custom_UNet_V1(nn.Module):
    def __init__(self,input_class, num_classes):
        super().__init__()

        self.enc1 = DoubleConv(in_channels=input_class,out_channels=16) 
        self.res1 = ResPath(16,2,1.0)
        self.down1 = Down(64)
        
        self.enc2 = DoubleConv(64,32)
        self.res2 = ResPath(32,2,1.0)
        self.down2 = Down(128)
        
        self.enc3 = DoubleConv(128,64)
        self.res3 = ResPath(64,2,1.0)
        self.down3 = Down(256)
        

        self.enc4 = DoubleConv(256,128)
        self.res4 = ResPath(128,2,1.0)
        self.down4 = Down(512)
        
        
        self.enc5 = Bottom(512,512)
        self.dim5 = nn.Conv2d(512,1,1)

        self.up4 = Up(128)
        self.dec4 = DoubleConv(256,256)
        self.dim4 = nn.Conv2d(256,1,1)

        self.up3 = Up(64)
        self.dec3 = DoubleConv(128,128)
        self.dim3 = nn.Conv2d(128,1,1)

        self.up2 = Up(32)
        self.dec2 = DoubleConv(64,64)
        self.dim2 = nn.Conv2d(64,1,1)

        self.up1 = Up(16)
        self.dec1 = DoubleConv(32,16)

        self.out = Outc(16,num_classes)



    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.down1((x1)))
        x3 = self.enc3(self.down2((x2)))
        x4 = self.enc4(self.down3((x3)))
        x5 = self.enc5(self.down4((x4)))

        x6 = self.dec4(torch.cat([(self.up4(x5)),self.res4(x4)],dim=1))
        x7 = self.dec3(torch.cat([(self.up3(x6)),self.res3(x3)],dim=1))
        x8 = self.dec2(torch.cat([(self.up2(x7)),self.res2(x2)],dim=1))

        x9 = self.dec1(torch.cat([(self.up1(x8)),self.res1(x1)],dim=1))

        out = self.out(x9)

        return [out,self.dim2(x8),self.dim3(x7),self.dim4(x6),self.dim5(x5)]
    


# sample = torch.randn((1,1,512,512))
# model = custom_UNet_V1(1,1)
# print(model(sample)[3].shape)



