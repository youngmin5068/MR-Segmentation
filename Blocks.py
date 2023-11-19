from torch.autograd import Variable, Function
import torch
from torch import nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
from torch.autograd import Variable
from torchvision.ops import deform_conv2d
import torch.nn.functional as F

'''
Deform Block
'''

class customDeformConv(nn.Module):
    def __init__(self, ci, co, kernel_size=3, stride=1, padding=1):
        super(customDeformConv, self).__init__()
        
        self.stride = stride
        self.padding = padding
        
        # 가중치 초기화
        self.weight = nn.Parameter(torch.Tensor(co, ci, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.Tensor(co))
        
        # 오프셋 및 마스크 초기화
        self.offset = nn.Conv2d(ci, 2 * kernel_size * kernel_size, kernel_size=kernel_size, stride=stride, padding=padding)
        self.mask = nn.Conv2d(ci, kernel_size * kernel_size, kernel_size=kernel_size, stride=stride, padding=padding)
        
        # 가중치와 바이어스 초기화
        nn.init.kaiming_uniform_(self.weight, a=0)
        nn.init.constant_(self.bias, 0)
        
        # 오프셋 초기화
        nn.init.constant_(self.offset.weight, 0)
        nn.init.constant_(self.offset.bias, 0)
        
        # 마스크 초기화 후 시그모이드 적용
        nn.init.constant_(self.mask.weight, 0)
        nn.init.constant_(self.mask.bias, 0)
        
    def forward(self, x):
        offset = self.offset(x)
        mask = torch.sigmoid(self.mask(x))  # 마스크를 0과 1 사이로 제한
        
        out = deform_conv2d(x, offset, self.weight, self.bias, stride=self.stride, padding=self.padding, mask=mask)
        
        return out



'''
LKA Block
'''

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x
    

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x




class LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)


    def forward(self, x):
        u = x.clone()        
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn


class Attention(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LKA(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class LKA_Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0.,drop_path=0., act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.GroupNorm(dim,dim)
        self.attn = Attention(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = nn.GroupNorm(dim,dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2            
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class TopTPercentChannelGate(nn.Module):
    def __init__(self, gate_channels, percent_t=1.0, reduction_ratio=16, pool_types=['avg', 'max']):
        super(TopTPercentChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.percent_t = percent_t
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types
        
    def forward(self, x):
        b, c, _, _ = x.size()
        x_flatten = x.view(b, c, -1)
        top_t = int(round(x_flatten.size(2) * self.percent_t))
        
        channel_att_sum = None

        for pool_type in self.pool_types:
            if pool_type == 'avg':
                selected_values, _ = x_flatten.topk(top_t, dim=2)
                pool = selected_values.mean(dim=2, keepdim=True)
            elif pool_type == 'max':
                selected_values, _ = x_flatten.topk(top_t, dim=2)
                pool = selected_values.max(dim=2, keepdim=True)[0]
            else:
                raise ValueError("Invalid pool_type, choose between 'avg' and 'max'")
                
            channel_att_raw = self.mlp(pool)
            
            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw
                
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)

        return x * scale
    

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
        
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale


class Topt_CBAM(nn.Module):
    def __init__(self, gate_channels, percent_t, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(Topt_CBAM, self).__init__()
        self.percent_t = percent_t
        self.reduction_ratio = reduction_ratio
        self.ChannelGate = TopTPercentChannelGate(gate_channels, percent_t=self.percent_t,reduction_ratio=self.reduction_ratio, pool_types=['avg', 'max'])
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out

    

# sample = torch.randn((1,1,512,512))
# model = DeformBlock(1)
# print(model(sample).shape)




class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

class custom_LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        #self.conv_spatial1 = nn.Conv2d(2, dim, 5, stride=1, padding=6, dilation=3)

        self.conv_spatial1 = nn.Conv2d(2, dim, 5, stride=1,padding=2)
        self.conv_spatial2 = nn.Conv2d(dim, dim, 7, stride=1, padding=3)
        self.norm = nn.GroupNorm(dim,dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)

        self.act = nn.Sigmoid()

    def forward(self, x):
        u = x.clone()
        avg_out = torch.mean(x,dim=1, keepdim=True)
        max_out,_ = torch.max(x,dim=1, keepdim=True)   
        x = torch.cat([avg_out, max_out],dim=1)
        
        x = self.conv_spatial1(x)
        x = self.conv_spatial2(x)
        x = self.norm(x)
        x = self.conv1(x)
        x = self.act(x)

        return x * u

