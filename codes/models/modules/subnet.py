import torch
import torch.nn as nn
import torch.nn.functional as F
import models.modules.utils as utils
from models.modules.RCAN import RCAB
from models.modules.common import default_conv
class CALayer(nn.Module):
    def __init__(self, channel, reduction=2):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=32, bias=True, CA=False):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(channel_in + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(channel_in + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(channel_in + 4 * gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        self.CA = CA
        if CA:
            self.CA_layer = CALayer(channel_out)

        if init == 'xavier':
            utils.initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        else:
            utils.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        utils.initialize_weights(self.conv5, 0)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        if self.CA:
            x5 = self.CA_layer(x5)

        return x5

class RCABlock(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', n_feat=64, reduction=16, n_resblocks=2, bias=True):
        super(RCABlock, self).__init__()
        
        self.in_conv = nn.Conv2d(channel_in, n_feat, 3, 1, 1, bias=bias)
        modules_body = [RCAB(default_conv, n_feat, 3, reduction, act=nn.LeakyReLU(negative_slope=0.2, inplace=True))
            for _ in range(n_resblocks)]
        self.rcabs = nn.Sequential(*modules_body)
        self.out_conv = nn.Conv2d(n_feat, channel_out, 3, 1, 1, bias=bias)
        if init == 'xavier':
            self.apply(utils.init_weights_xavier)
        else:
            self.apply(utils.init_weights)
    def forward(self, x):
        x = self.in_conv(x)
        res = self.rcabs(x)
        res += x
        res = self.out_conv(res)
        return res

def subnet(net_structure, init='xavier', CA=False):
    def constructor(channel_in, channel_out):
        if net_structure == 'DBNet':
            return DenseBlock(channel_in, channel_out, init)
        elif net_structure == 'DBNet_CA':
            return DenseBlock(channel_in, channel_out, init, CA=True)
        elif net_structure == 'RCABlock':
            return RCABlock(channel_in, channel_out, init)
        else:
            return None

    return constructor
