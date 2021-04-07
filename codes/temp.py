import torch
from models.modules.INN import InvRescaleNet
from models.modules.subnet import subnet
from collections import OrderedDict
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
INN = InvRescaleNet(
            channel_in=3, channel_out=3, subnet_constructor=subnet('DBNet'), 
            block_num=[8, 8], downscale_trainable=True).cuda()

load_path = '/data/xiongjianyu/myINNSR/experiments/train_INN_x4_trainable_Haar/models/275000_INN.pth'
load_net = torch.load(load_path)
load_net_clean = OrderedDict()  # remove unnecessary 'module.'
for k, v in load_net.items():
    if k.startswith('module.'):
        load_net_clean[k[7:]] = v
    else:
        load_net_clean[k] = v
INN.load_state_dict(load_net_clean, strict=False)
for i in range(2):
    print(INN.operations[i * 9].a)
    print(INN.operations[i * 9].b)
    print(INN.operations[i * 9].c)
    print(INN.operations[i * 9].d)