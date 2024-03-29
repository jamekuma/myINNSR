import logging
from collections import OrderedDict

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
from models.modules.INN import InvRescaleNet
from models.modules.RCAN import RCAN
from models.modules.subnet import subnet
from models.modules.loss import ReconstructionLoss
# m = Laplace(torch.tensor([0.0]), torch.tensor([1.0]))
logger = logging.getLogger('base')

class INNSRModel(BaseModel):
    '''
    重头开始训练
    '''
    def __init__(self, opt):
        super(INNSRModel, self).__init__(opt)

        train_opt = opt['train']
        test_opt = opt['test']
        self.train_opt = train_opt
        self.test_opt = test_opt

        INN_network_opt = opt['network']['INN']
        Gap_network_opt = opt['network']['GapNN']
        self.INN_network_opt = INN_network_opt
        self.Gap_network_opt = Gap_network_opt
        self.INN = InvRescaleNet(
            channel_in=INN_network_opt['in_nc'], channel_out=INN_network_opt['out_nc'],
             subnet_constructor=subnet('DBNet'), block_num=INN_network_opt['block_num']).to(self.device)
        self.INN = DataParallel(self.INN)
        self.GapNN = RCAN(n_resgroups=Gap_network_opt['n_resgroups'], n_resblocks=Gap_network_opt['n_resblocks']).to(self.device)
        self.GapNN = DataParallel(self.GapNN)
        
        # print network
        self.print_network()
        self.load()
        self.Quantization = Quantization()

        if self.is_train:
            if not INN_network_opt['fixed']:
                self.INN.train()
            if not Gap_network_opt['fixed']:
                self.GapNN.train()

            # loss
            self.Reconstruction_forw = ReconstructionLoss(train_opt['pixel_criterion_forw'])
            self.Reconstruction_back = ReconstructionLoss(train_opt['pixel_criterion_back'])
            self.Reconstruction_Gap = ReconstructionLoss(train_opt['pixel_criterion_Gap'])
 
            # optimizers
            wd_INN = train_opt['weight_decay_INN'] if train_opt['weight_decay_INN'] else 0
            wd_GapNN = train_opt['weight_decay_GapNN'] if train_opt['weight_decay_GapNN'] else 0
            optim_params = []   # 需要优化的参数们
            if INN_network_opt['fixed']:
                for p in self.INN.parameters():
                    p.requires_grad = False
            if Gap_network_opt['fixed']:
                for p in self.GapNN.parameters():
                    p.requires_grad = False
                    
            # if not INN_network_opt['fixed']:
            for k, v in self.INN.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                # else:
                #     logger.warning('Params [{:s}] will not optimize.'.format(k))
            
            # if not Gap_network_opt['fixed']:
            for k, v in self.GapNN.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                # else:
                #     logger.warning('Params [{:s}] will not optimize.'.format(k))
            
            for m in self.GapNN.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.xavier_uniform_(m.weight)
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
        out_y = forw_out[:, :3, :, :]
        out_z = forw_out[:, 3:, :, :]
        l_forw_fit = self.train_opt['lambda_fit_forw'] * self.Reconstruction_forw(out_y, ref_L)

        
        out_z = out_z.reshape([forw_out.shape[0], -1])
        l_forw_ce = self.train_opt['lambda_ce_forw'] * torch.sum(out_z**2) / out_z.shape[0]

        return l_forw_fit, l_forw_ce

    def INN_loss_backward(self, back_out):
        back_out_image = back_out[:, :3, :, :]
        l_back_rec = self.train_opt['lambda_rec_back'] * self.Reconstruction_back(self.real_H, back_out_image)

        return l_back_rec
    
    def GapNN_loss(self, forw_out, GapNN_out):
        return self.Reconstruction_Gap(GapNN_out, forw_out[:, :3, :, :])


    def optimize_parameters(self, step):
        INN_network_opt = self.opt['network']['INN']
        Gap_network_opt = self.opt['network']['GapNN']
        if not INN_network_opt['fixed']:
            self.INN.train()
        if not Gap_network_opt['fixed']:
            self.GapNN.train()
        self.optimizer.zero_grad()

        ########## 正向loss
        # with torch.no_grad():
        GapNN_out = self.GapNN(self.ref_L)
        self.forw_out = self.INN(self.real_H) # 正向推理
        l_forw_fit, l_forw_ce = self.INN_loss_forward(self.forw_out, GapNN_out)

        zshape = self.forw_out[:, 3:, :, :].shape

        ######### 逆向loss
        back_out = self.INN(torch.cat((GapNN_out, self.gaussian_batch(zshape)), dim=1), rev=True)
        l_back_rec = self.INN_loss_backward(back_out)

        
        l_gap = self.GapNN_loss(self.forw_out, GapNN_out)  # 在LR域的Gap
        # total loss
        loss = 0
        loss += l_forw_fit + l_back_rec + l_forw_ce
        # loss += l_gap
        # loss += l_HR
        # print('l_back_rec: {:.4e}'.format(l_back_rec))
        loss.backward()

        # gradient clipping
        if self.train_opt['gradient_clipping']:
            nn.utils.clip_grad_norm_(self.INN.parameters(), self.train_opt['gradient_clipping'])
            nn.utils.clip_grad_norm_(self.GapNN.parameters(), self.train_opt['gradient_clipping'])

        self.optimizer.step()

        # set log
        self.log_dict['l_forw_fit'] = l_forw_fit.item()
        self.log_dict['l_forw_ce'] = l_forw_ce.item()
        self.log_dict['l_back_rec'] = l_back_rec.item()
        self.log_dict['l_gap'] = l_gap.item()

    def test(self):
        Lshape = self.ref_L.shape

        input_dim = Lshape[1]
        self.input = self.real_H

        zshape = [Lshape[0], input_dim * (self.opt['scale']**2) - Lshape[1], Lshape[2], Lshape[3]]

        gaussian_scale = 1
        if self.test_opt and self.test_opt['gaussian_scale'] != None:
            gaussian_scale = self.test_opt['gaussian_scale']

        self.INN.eval()
        self.GapNN.eval()
        with torch.no_grad():
            self.forw_L = self.INN(x=self.input)[:, :3, :, :]
            self.forw_L = self.Quantization(self.forw_L)
            self.Gap_L = self.GapNN(self.ref_L)
            y_forw = torch.cat((self.Gap_L, gaussian_scale * self.gaussian_batch(zshape)), dim=1)
            self.fake_H = self.INN(x=y_forw, rev=True)
            # Reconstruction_back = ReconstructionLoss('l1')
            # l_back_rec = 10 * Reconstruction_back(self.real_H, self.fake_H)
            # logger.info('l_back_rec: {:.4e}'.format(l_back_rec))

        INN_network_opt = self.opt['network']['INN']
        Gap_network_opt = self.opt['network']['GapNN']
        if not INN_network_opt['fixed']:
            self.INN.train()
        if not Gap_network_opt['fixed']:
            self.GapNN.train()

    # def downscale(self, HR_img):
    #     self.netG.eval()
    #     with torch.no_grad():
    #         LR_img = self.netG(x=HR_img)[:, :3, :, :]
    #         LR_img = self.Quantization(self.forw_L)
    #     self.netG.train()

    #     return LR_img

    def upscale(self, LR_img, scale, gaussian_scale=1):
        Lshape = LR_img.shape
        zshape = [Lshape[0], Lshape[1] * (scale**2 - 1), Lshape[2], Lshape[3]]
       

        self.INN.eval()
        self.GapNN.eval()
        with torch.no_grad():
            y_ = torch.cat((self.GapNN(LR_img), gaussian_scale * self.gaussian_batch(zshape)), dim=1)
            HR_img = self.INN(x=y_, rev=True)[:, :3, :, :]
        self.INN.train()
        self.GapNN.train()

        return HR_img

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['LR_ref'] = self.ref_L.detach()[0].float().cpu()
        out_dict['SR'] = self.fake_H.detach()[0].float().cpu()
        out_dict['LR_forw'] = self.forw_L.detach()[0].float().cpu()
        out_dict['LR_gap'] = self.Gap_L.detach()[0].float().cpu()
        out_dict['GT'] = self.real_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.INN)

        net_struc_str = '{} - {}'.format(self.INN.__class__.__name__,
                                             self.INN.module.__class__.__name__)
        logger.info('Network INN structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

        s, n = self.get_network_description(self.GapNN)
        net_struc_str = '{} - {}'.format(self.GapNN.__class__.__name__,
                                             self.GapNN.module.__class__.__name__)
        logger.info('Network GapNN structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    def load(self):
        load_path_INN = self.opt['path']['pretrain_INN']
        if load_path_INN is not None:
            logger.info('Loading model for INN [{:s}] ...'.format(load_path_INN))
            if isinstance(self.INN, nn.DataParallel) or isinstance(self.INN, DistributedDataParallel):
                self.load_network(load_path_INN, self.INN.module, self.opt['path']['strict_load'])
            else:
                self.load_network(load_path_INN, self.INN, self.opt['path']['strict_load'])

        load_path_GapNN = self.opt['path']['pretrain_GapNN']
        if load_path_GapNN is not None:
            logger.info('Loading model for GapNN [{:s}] ...'.format(load_path_GapNN))
            if isinstance(self.GapNN, nn.DataParallel) or isinstance(self.GapNN, DistributedDataParallel):
                self.load_network(load_path_GapNN, self.GapNN.module, self.opt['path']['strict_load'])
            else:
                self.load_network(load_path_GapNN, self.GapNN, self.opt['path']['strict_load'])

    def save(self, iter_label):
        if not self.INN_network_opt['fixed']:
            self.save_network(self.INN, 'INN', iter_label)
        if not self.Gap_network_opt['fixed']:
            self.save_network(self.GapNN, 'GapNN', iter_label)

    # def resume_training(self, resume_state):
    #     super().resume_training(resume_state)
    #     self.optimizer = self.optimizers[0]