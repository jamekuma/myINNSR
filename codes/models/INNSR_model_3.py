import logging
from collections import OrderedDict
import math
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
# import models.networks as networks
# import models.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn.modules import L1Loss, MSELoss
from .base_model import BaseModel
# from models.modules.loss import ReconstructionLoss
from models.modules.Quantization import Quantization
from torch.distributions.laplace import Laplace
from models.modules.INN import InvRescaleNet, InvSRNet, InvSRNet_2
from models.modules.RCAN import RCAN
from models.modules.subnet import subnet
from models.modules.loss import ReconstructionLoss
# m = Laplace(torch.tensor([0.0]), torch.tensor([1.0]))
logger = logging.getLogger('base')

class INNSRModel(BaseModel):
    '''
    重头开始训练, 原始版本，不带GapNN
    '''
    def __init__(self, opt, print_network=True):
        super(INNSRModel, self).__init__(opt)

        train_opt = opt['train']
        test_opt = opt['test']
        self.train_opt = train_opt
        self.test_opt = test_opt

        INN_network_opt = opt['network']['INN']
        self.INN_network_opt = INN_network_opt
        # self.INN = InvSRNet(
        #     channel_in=INN_network_opt['in_nc'], channel_out=INN_network_opt['out_nc'],
        #      subnet_constructor=subnet(INN_network_opt['subnet_type']), block_num=INN_network_opt['block_num'],
        #      downscale_trainable=INN_network_opt['downscale_trainable'], down_num=int(math.log(opt['scale'], 2))).to(self.device)
        self.INN = InvSRNet_2(
            channel_in=INN_network_opt['in_nc'], channel_out=INN_network_opt['out_nc'],n_feat=INN_network_opt['n_feat'],
             subnet_constructor=subnet(INN_network_opt['subnet_type']), block_num=INN_network_opt['block_num'], scale=opt['scale']).to(self.device)
        self.INN = DataParallel(self.INN)
        
        # print 
        if print_network:
            self.print_network()
        self.load()
        self.Quantization = Quantization()

        if self.is_train:
            if not INN_network_opt['fixed']:
                self.INN.train()

            # loss
            self.Reconstruction_forw = ReconstructionLoss(train_opt['pixel_criterion_forw'])
            self.Reconstruction_back = ReconstructionLoss(train_opt['pixel_criterion_back'])
            self.Reconstruction_Gap = ReconstructionLoss(train_opt['pixel_criterion_Gap'])
 
            # optimizers
            wd_INN = train_opt['weight_decay_INN'] if train_opt['weight_decay_INN'] else 0
            optim_params = []   # 需要优化的参数们
            if INN_network_opt['fixed']:
                for p in self.INN.parameters():
                    p.requires_grad = False
                    
            # if not INN_network_opt['fixed']:
            for k, v in self.INN.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
            
            # print(train_opt['beta1'], train_opt['beta2'])
            self.optimizer = torch.optim.Adam(optim_params, lr=train_opt['lr'],
                                                weight_decay=wd_INN,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer)

            # MultiStepLR
            for optimizer in self.optimizers:
                self.schedulers.append(MultiStepLR(optimizer, milestones=train_opt['lr_steps'], gamma=train_opt['lr_gamma']))

            self.log_dict = OrderedDict()

    def feed_data(self, data):
        self.ref_L = data['LQ'].to(self.device)  # LQ
        self.real_H = data['GT'].to(self.device)  # GT


    def gaussian_batch(self, dims):
        res = torch.randn(tuple(dims)).to(self.device)
        return res

    def INN_loss_forward(self, forw_out, ref_L):
        l_forw_fit = self.train_opt['lambda_fit_forw'] * self.Reconstruction_forw(forw_out, ref_L)
        return l_forw_fit

    def INN_loss_backward(self, back_out):
        back_out_image = back_out[:, :3, :, :]
        l_back_rec = self.train_opt['lambda_rec_back'] * self.Reconstruction_back(self.real_H, back_out_image)

        return l_back_rec
    


    def optimize_parameters(self, step):
        self.optimizer.zero_grad()

        ######### 逆向loss
        back_out = self.INN(self.ref_L, rev=True)
        l_back_rec = self.INN_loss_backward(back_out)
        sr_image = back_out[:, :3, :, :]
        ########## 正向loss
        # with torch.no_grad():
        if self.train_opt['lambda_fit_forw'] != 0:
            self.forw_out = self.INN(sr_image) # 正向推理
            l_forw_fit = self.INN_loss_forward(self.forw_out, self.ref_L)
        else:
            l_forw_fit = torch.Tensor([0]).to(self.device)


        

        # total loss
        loss = 0
        loss += l_forw_fit + l_back_rec
        # loss += l_gap
        # loss += l_HR
        loss.backward()

        # gradient clipping
        if self.train_opt['gradient_clipping']:
            nn.utils.clip_grad_norm_(self.INN.parameters(), self.train_opt['gradient_clipping'])

        self.optimizer.step()

        # set log
        self.log_dict['l_forw_fit'] = l_forw_fit.item()
        # self.log_dict['l_forw_ce'] = l_forw_ce.item()
        self.log_dict['l_back_rec'] = l_back_rec.item()

    def test(self):
        Lshape = self.ref_L.shape

        self.input = self.real_H

        zshape = [Lshape[0], Lshape[1] * (self.opt['scale']**2) - Lshape[1], Lshape[2], Lshape[3]]

        gaussian_scale = 1
        if self.test_opt and self.test_opt['gaussian_scale'] != None:
            gaussian_scale = self.test_opt['gaussian_scale']

        self.INN.eval()
        with torch.no_grad():
            self.forw_L = self.INN(x=self.input)[:, :3, :, :]
            self.forw_L = self.Quantization(self.forw_L)
            self.SR_img = self.INN(x=self.ref_L, rev=True)[:, :3, :, :]
            self.rescaling_img = self.INN(x=self.forw_L, rev=True)[:, :3, :, :]
        INN_network_opt = self.opt['network']['INN']
        if not INN_network_opt['fixed']:
            self.INN.train()

    # def downscale(self, HR_img):
    #     self.netG.eval()
    #     with torch.no_grad():
    #         LR_img = self.netG(x=HR_img)[:, :3, :, :]
    #         LR_img = self.Quantization(self.forw_L)
    #     self.netG.train()

    #     return LR_img

    def upscale(self, LR_img, scale, gaussian_scale=1, gaussian_samples=None):
        Lshape = LR_img.shape
        zshape = [Lshape[0], Lshape[1] * (scale**2 - 1), Lshape[2], Lshape[3]]
        LR_img = LR_img.to(self.device)
        with torch.no_grad():
            HR_img = self.INN(x=LR_img, rev=True)[:, :3, :, :]
        self.INN.train()
        return HR_img

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['LR_ref'] = self.ref_L.detach()[0].float().cpu()
        out_dict['SR'] = self.SR_img.detach()[0].float().cpu()
        out_dict['LR_forw'] = self.forw_L.detach()[0].float().cpu()
        out_dict['rescaling'] = self.rescaling_img.detach()[0].float().cpu()
        out_dict['GT'] = self.real_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.INN)

        net_struc_str = '{} - {}'.format(self.INN.__class__.__name__,
                                             self.INN.module.__class__.__name__)
        logger.info('Network INN structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    def load(self):
        load_path_INN = self.opt['path']['pretrain_INN']
        if load_path_INN is not None:
            logger.info('Loading model for INN [{:s}] ...'.format(load_path_INN))
            self.load_network(load_path_INN, self.INN, self.opt['path']['strict_load'])

    def save(self, iter_label):
        if not self.INN_network_opt['fixed']:
            self.save_network(self.INN, 'INN', iter_label)
