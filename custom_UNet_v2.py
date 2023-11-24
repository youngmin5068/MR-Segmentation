import torch
import torch.nn as nn
import torch.nn.functional as F
from deform2d import DeformableAttention2D
from Blocks import LKA_custom_v2, Topt_CBAM


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,groups=in_channels),
            nn.Conv2d(in_channels,2*in_channels,1),

            nn.GroupNorm(2*in_channels,2*in_channels),

            nn.Conv2d(2*in_channels, 2*in_channels, kernel_size=3, padding=1, groups=2*in_channels),
            nn.Conv2d(2*in_channels,out_channels,1),

            nn.GELU()
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class PixelUnshuffle_Down(nn.Module):
    def __init__(self, in_channels, out_channels, down_scale = 2):
        super().__init__()
        self.down = nn.PixelUnshuffle(downscale_factor=down_scale)
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=3, padding=1)

    def forward(self,x):
        x = self.down(x)
        x = self.conv(x)

        return x
    

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()


        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
class PixelShuffle_Up(nn.Module):
    def __init__(self, px_in,px_out, in_channels, out_channels, up_scale = 2):
        super().__init__()
        self.up = nn.PixelShuffle(upscale_factor=up_scale)
        self.conv1 = nn.Conv2d(px_in,px_out,kernel_size=1)
 
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=3, padding=1)

    def forward(self,x1,x2):
        x1 = self.up(x1)
        x1 = self.conv1(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class attentionGate(nn.Module):
    def __init__(self,dim,percent_t):
        super().__init__()
        self.channelGate = Topt_CBAM(dim,percent_t)
        self.lka = LKA_custom_v2(dim)
    def forward(self, x):
        u = x.clone()
        x = self.channelGate(x)
        x = self.lka(x) + u
        return x


class ResPath(torch.nn.Module):
    """
    Implements ResPath-like modified skip connection

    """

    def __init__(self, in_chnls, n_lvl,percent_t=1.0):
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
    


class custom_UNet_V2(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(custom_UNet_V2, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear


        self.attn1 = ResPath(in_chnls=16,n_lvl=4)
        self.attn2 = ResPath(in_chnls=32,n_lvl=3)
        self.attn3 = ResPath(in_chnls=64,n_lvl=2)
        self.attn4 = ResPath(in_chnls=128,n_lvl=1)

        self.inc = (DoubleConv(n_channels, 16)) 
        self.res0 = ResPath(16,2)

        #maxpool
        self.down1 = (Down(16, 32))
        self.down2 = (Down(32, 64)) 
        #subpixel down
        self.down3 = (PixelUnshuffle_Down(256, 128))
        self.down4 = (PixelUnshuffle_Down(512, 256)) 


        #subpixel up
        self.up1 = (PixelShuffle_Up(64,128,256,128)) 
        self.up2 = (PixelShuffle_Up(32,64,128,64)) 


        #upsample
        self.up3 = (Up(64, 32))
        self.up4 = (Up(32, 16))
        self.outc = (OutConv(16, n_classes))

    def forward(self, x):
        x1 = self.inc(x) #32,512,512
        x2 = self.down1(x1) #64,256,256
        x3 = self.down2(x2) #128,128,128
        x4 = self.down3(x3) #256,64,64
        x5 = self.down4(x4) #512,32,32


        x6 = self.up1(x5,self.attn4(x4))
        x7 = self.up2(x6, self.attn3(x3))
        x8 = self.up3(x7, self.attn2(x2))
        x9 = self.up4(x8, self.attn1(x1))
        
        logits = self.outc(x9)


        return logits


class custom_UNet_V2_small(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(custom_UNet_V2_small, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear


        self.attn1 = ResPath(in_chnls=16,n_lvl=4,percent_t=0.1)
        self.attn2 = ResPath(in_chnls=32,n_lvl=3,percent_t=0.1)
        self.attn3 = ResPath(in_chnls=64,n_lvl=2,percent_t=0.25)
        self.attn4 = ResPath(in_chnls=128,n_lvl=1,percent_t=1.0)

        self.inc = (DoubleConv(n_channels, 16)) 
        self.res0 = ResPath(16,2)

        #maxpool
        self.down1 = (Down(16, 32))
        self.down2 = (Down(32, 64)) 
        #subpixel down
        self.down3 = (PixelUnshuffle_Down(256, 128))
        self.down4 = (PixelUnshuffle_Down(512, 256)) 


        #subpixel up
        self.up1 = (PixelShuffle_Up(64,128,256,128)) 
        self.up2 = (PixelShuffle_Up(32,64,128,64)) 


        #upsample
        self.up3 = (Up(64, 32))
        self.up4 = (Up(32, 16))
        self.outc = (OutConv(16, n_classes))

    def forward(self, x):
        x1 = self.inc(x) #32,512,512
        x2 = self.down1(x1) #64,256,256
        x3 = self.down2(x2) #128,128,128
        x4 = self.down3(x3) #256,64,64
        x5 = self.down4(x4) #512,32,32


        x6 = self.up1(x5,self.attn4(x4))
        x7 = self.up2(x6, self.attn3(x3))
        x8 = self.up3(x7, self.attn2(x2))
        x9 = self.up4(x8, self.attn1(x1))
        
        logits = self.outc(x9)


        return logits
    


# sample = torch.randn((1,1,512,512))
# model = custom_UNet_V2(1,1)
# print(model(sample)[1].shape)
    