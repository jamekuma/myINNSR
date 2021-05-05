import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class InvBlockExp(nn.Module):
    '''可逆网络模块'''
    def __init__(self, subnet_constructor, channel_num, channel_split_num, clamp=1.):
        super(InvBlockExp, self).__init__()

        self.split_len1 = channel_split_num
        self.split_len2 = channel_num - channel_split_num

        self.clamp = clamp

        self.F = subnet_constructor(self.split_len2, self.split_len1)
        self.G = subnet_constructor(self.split_len1, self.split_len2)
        self.H = subnet_constructor(self.split_len1, self.split_len2)

    def forward(self, x, rev=False):
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

        if not rev:
            y1 = x1 + self.F(x2)
            self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
            y2 = x2.mul(torch.exp(self.s)) + self.G(y1)
        else:
            self.s = self.clamp * (torch.sigmoid(self.H(x1)) * 2 - 1)
            y2 = (x2 - self.G(x1)).div(torch.exp(self.s))
            y1 = x1 - self.F(y2)

        return torch.cat((y1, y2), 1)

    def jacobian(self, x, rev=False):
        if not rev:
            jac = torch.sum(self.s)
        else:
            jac = -torch.sum(self.s)

        return jac / x.shape[0]

class InvBlockExpPlus(nn.Module):
    '''可逆网络模块Plus'''
    def __init__(self, subnet_constructor, channel_num, channel_split_num, clamp=1.):
        super(InvBlockExpPlus, self).__init__()

        self.split_len1 = channel_split_num
        self.split_len2 = channel_num - channel_split_num

        self.clamp = clamp

        self.E = subnet_constructor(self.split_len2, self.split_len1)
        self.F = subnet_constructor(self.split_len2, self.split_len1)
        self.G = subnet_constructor(self.split_len1, self.split_len2)
        self.H = subnet_constructor(self.split_len1, self.split_len2)
        

    def forward(self, x, rev=False):
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

        if not rev:
            y1 = x1.mul(torch.exp(torch.sigmoid(self.E(x2)) * 2 - 1)) + self.F(x2)
            self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
            y2 = x2.mul(torch.exp(self.s)) + self.G(y1)
        else:
            self.s = self.clamp * (torch.sigmoid(self.H(x1)) * 2 - 1)
            y2 = (x2 - self.G(x1)).div(torch.exp(self.s))
            y1 = (x1 - self.F(y2)).div(torch.exp(torch.sigmoid(self.E(y2)) * 2 - 1))

        return torch.cat((y1, y2), 1)


class HaarDownsampling(nn.Module):
    '''
    哈尔小波变换
    '''
    def __init__(self, channel_in, trainable=False):
        super(HaarDownsampling, self).__init__()
        self.channel_in = channel_in
        
        self.a = torch.ones(1).cuda()
        self.b = torch.ones(1).cuda()
        self.c = torch.ones(1).cuda()
        self.d = -torch.ones(1).cuda()
        self.a = nn.Parameter(self.a)
        self.b = nn.Parameter(self.b)
        self.c = nn.Parameter(self.c)
        self.d = nn.Parameter(self.d)
        self.a.requires_grad = trainable
        self.b.requires_grad = trainable
        self.c.requires_grad = trainable
        self.d.requires_grad = trainable

        # haar_weights = nn.Parameter(haar_weights)
        # haar_weights.requires_grad = trainable

    def forward(self, x, rev=False):
        haar_weights = self.get_weight(rev)
        if not rev:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 4 * np.log(1/16.)

            out = F.conv2d(x, haar_weights, bias=None, stride=2, groups=self.channel_in) / ((self.a * self.d - self.b * self.c) ** 2)
            out = out.reshape([x.shape[0], self.channel_in, 4, x.shape[2] // 2, x.shape[3] // 2])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2] // 2, x.shape[3] // 2])
            return out
        else:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 4 * np.log(16.)

            out = x.reshape([x.shape[0], 4, self.channel_in, x.shape[2], x.shape[3]])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2], x.shape[3]])
            out = F.pixel_shuffle(out, 2)
            out = F.conv2d(out, haar_weights, bias=None, stride=2, groups=self.channel_in)
            out = F.pixel_shuffle(out, 2)
            return out

    def jacobian(self, x, rev=False):
        return self.last_jac

    def get_weight(self, rev=False):
        haar_weights = torch.ones(4 * self.channel_in, 1, 2, 2).cuda()
        for i in range(self.channel_in):
            haar_weights[4 * i, 0, 0, 0] = self.a ** 2 if not rev else self.d ** 2
            haar_weights[4 * i, 0, 0, 1] = self.a * self.b if not rev else -self.d * self.b
            haar_weights[4 * i, 0, 1, 0] = self.a * self.b if not rev else -self.d * self.b   #  1  1
            haar_weights[4 * i, 0, 1, 1] = self.b ** 2 if not rev else self.b ** 2            #  1  1


            haar_weights[4 * i + 1, 0, 0, 0] = self.a * self.c if not rev else  -self.d * self.c
            haar_weights[4 * i + 1, 0, 0, 1] = self.a * self.d if not rev else self.a * self.d
            haar_weights[4 * i + 1, 0, 1, 0] = self.b * self.c if not rev else self.c * self.b  #  1 -1
            haar_weights[4 * i + 1, 0, 1, 1] = self.b * self.d if not rev else -self.a * self.b #  1 -1

            haar_weights[4 * i + 2, 0, 0, 0] = self.a * self.c if not rev else -self.d * self.c
            haar_weights[4 * i + 2, 0, 0, 1] = self.b * self.c if not rev else self.c * self.b
            haar_weights[4 * i + 2, 0, 1, 0] = self.a * self.d if not rev else self.a * self.d      #  1  1
            haar_weights[4 * i + 2, 0, 1, 1] = self.b * self.d if not rev else -self.a * self.b     # -1 -1

            haar_weights[4 * i + 3, 0, 0, 0] = self.c ** 2 if not rev else self.c ** 2
            haar_weights[4 * i + 3, 0, 0, 1] = self.c * self.d if not rev else -self.a * self.c
            haar_weights[4 * i + 3, 0, 1, 0] = self.c * self.d if not rev else -self.a * self.c  #  1 -1
            haar_weights[4 * i + 3, 0, 1, 1] = self.d ** 2 if not rev else self.a ** 2           # -1  1
        # print(haar_weights)
        # haar_weights = torch.cat([haar_weights] * self.channel_in, 0)
        return haar_weights

class InvRescaleNet(nn.Module):
    '''
    整个可逆网络结构
    '''
    def __init__(self, channel_in=3, channel_out=3, subnet_constructor=None, block_num=[], down_num=2, downscale_trainable=False, plus=False):
        '''
        subnet_constructor: 学习函数的构造器, 参数为(输入通道数, 输出通道数)
        down_num: 下采样的次数, 即down_num = log(scale)
        '''
        super(InvRescaleNet, self).__init__()

        operations = []

        current_channel = channel_in
        for i in range(down_num):
            b = HaarDownsampling(current_channel, downscale_trainable)
            operations.append(b)
            current_channel *= 4
            for j in range(block_num[i]):
                if plus:
                    b = InvBlockExpPlus(subnet_constructor, current_channel, channel_out)
                else:
                    b = InvBlockExp(subnet_constructor, current_channel, channel_out)
                operations.append(b)

        self.operations = nn.ModuleList(operations)

    def forward(self, x, rev=False, cal_jacobian=False):
        out = x
        jacobian = 0

        if not rev:
            for op in self.operations:
                out = op.forward(out, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)
        else:
            for op in reversed(self.operations):
                out = op.forward(out, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)

        if cal_jacobian:
            return out, jacobian
        else:
            return out


class InvSRNet(nn.Module):
    '''
    INNSR_model_3
    '''
    def __init__(self, channel_in=3, channel_out=3, subnet_constructor=None, block_num=[], down_num=2, downscale_trainable=False):
        super(InvSRNet, self).__init__()
        self.down_num = down_num
        current_channel = channel_in
        self.upscale_blocks = nn.ModuleList()
        self.downscale_blocks = nn.ModuleList()
        self.inv_blocks = nn.ModuleList()
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        for i in range(down_num):
            self.downscale_blocks.append(nn.Conv2d(current_channel, current_channel * 4, 3, stride=2, padding=1))
            operations = []
            for j in range(block_num[i]):
                operations.append(InvBlockExp(subnet_constructor, current_channel * 4, channel_out))
            self.inv_blocks.append(nn.ModuleList(operations))
            self.upscale_blocks.append(
                    nn.Sequential(
                        nn.PixelShuffle(2),
                    )
                )
            current_channel *= 4
        self.end_forward_conv = nn.Conv2d(current_channel, channel_out, 3, stride=1, padding=1)
        self.end_backward_conv = nn.Conv2d(channel_out, current_channel, 3, stride=1, padding=1)
    def forward(self, x, rev=False):
        out = x
        if not rev:
            for i in range(self.down_num):
                out = self.lrelu(self.downscale_blocks[i](out))
                for inv in self.inv_blocks[i]:
                    out = inv(out)
            out = self.lrelu(self.end_forward_conv(out))
        else:
            out = self.lrelu(self.end_backward_conv(out))
            for i in reversed(range(self.down_num)):
                for inv in reversed(self.inv_blocks[i]):
                    out = inv(out, rev=True)
                out = self.upscale_blocks[i](out)
        return out
