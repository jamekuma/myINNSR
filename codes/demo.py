import os.path as osp
import logging
import time
import torch 
import argparse
from collections import OrderedDict
from data.dataset import LQGTDataset
import numpy as np
import config.config as config
import utils.utils as util

from models.INNSR_model_0 import INNSRModel as M
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, required=True, help='Path to options YMAL file.')
opt = config.parse(parser.parse_args().opt, is_train=False)
opt = config.dict_to_nonedict(opt)
# opt['path']['output_root'] = osp.join(opt['datasets']['test_1']['dataroot_GT'], 'results')
opt['path']['log'] = opt['path']['output_root'] 

util.mkdirs(opt['path']['output_root'])
util.setup_logger('base', opt['path']['log'], 'test_' + opt['name'], level=logging.INFO,
                  screen=True, tofile=True)
logger = logging.getLogger('base')
# logger.info(config.dict2str(opt))


model = M(opt, print_network=False)
dataset_opt = opt['datasets']['test_1']
test_set = LQGTDataset(dataset_opt)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1,
                                           pin_memory=True)
logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))

test_set_name = test_loader.dataset.opt['name']
logger.info('\nTesting [{:s}]...'.format(test_set_name))
test_start_time = time.time()
dataset_dir = osp.join(opt['path']['output_root'], test_set_name)
util.mkdir(dataset_dir)

# gaussian_samples=[48, 64, 80]

for data in test_loader:
    model.feed_data(data)
    img_path = data['GT_path'][0]
    img_name = osp.splitext(osp.basename(img_path))[0]

    # model.test()
    # visuals = model.get_current_visuals()
    
    lr_img = util.tensor2img(data['LQ'][0])
    gt_img = util.tensor2img(data['GT'][0])  # uint8
    sr_img = util.tensor2img(model.upscale(data['LQ'], 4)[0])
    
    util.save_img(lr_img, osp.join(dataset_dir, img_name + '_LR.png'))
    util.save_img(gt_img, osp.join(dataset_dir, img_name + '_GT.png'))
    util.save_img(sr_img, osp.join(dataset_dir, img_name + '_SR.png'))

    # calculate PSNR and SSIM
    gt_img = gt_img / 255.
    sr_img = sr_img / 255.

    crop_border = opt['crop_border'] if opt['crop_border'] else opt['scale']
    if crop_border == 0:
        cropped_sr_img = sr_img
        cropped_gt_img = gt_img
    else:
        cropped_sr_img = sr_img[crop_border:-crop_border, crop_border:-crop_border, :]
        cropped_gt_img = gt_img[crop_border:-crop_border, crop_border:-crop_border, :]

    psnr = util.calculate_psnr(cropped_sr_img * 255, cropped_gt_img * 255)
    ssim = util.calculate_ssim(cropped_sr_img * 255, cropped_gt_img * 255)


    if gt_img.shape[2] == 3:  # RGB image
        sr_img_y = util.bgr2ycbcr(sr_img, only_y=True)
        gt_img_y = util.bgr2ycbcr(gt_img, only_y=True)
        if crop_border == 0:
            cropped_sr_img_y = sr_img_y
            cropped_gt_img_y = gt_img_y
        else:
            cropped_sr_img_y = sr_img_y[crop_border:-crop_border, crop_border:-crop_border]
            cropped_gt_img_y = gt_img_y[crop_border:-crop_border, crop_border:-crop_border]
        psnr_y = util.calculate_psnr(cropped_sr_img_y * 255, cropped_gt_img_y * 255)
        ssim_y = util.calculate_ssim(cropped_sr_img_y * 255, cropped_gt_img_y * 255)

        logger.info(
                '{:20s} - PSNR: {:.6f} dB; SSIM: {:.6f}; PSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}.'.
            format(img_name, psnr, ssim, psnr_y, ssim_y))
        
    else:
        logger.info('{:20s} - PSNR: {:.6f} dB; SSIM: {:.6f}.'.format(img_name, psnr, ssim))
