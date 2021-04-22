import os.path as osp
import logging
import time
import torch
import argparse
from collections import OrderedDict
from data.dataset import LQGTDataset
import numpy as np
import os
import config.config as config
import utils.utils as util

#### options
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, required=True, help='Path to options YMAL file.')
parser.add_argument('-gpu', type=str, help='gpu_id', default='0')
parser.add_argument('--debug', help='debug mode, do not make any dict', action="store_true")
args = parser.parse_args()
opt = config.parse(args.opt, is_train=False)
opt = config.dict_to_nonedict(opt)

opt['gpu_ids'] = [int(x) for x in args.gpu.split(',')]
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
print('export CUDA_VISIBLE_DEVICES=' + args.gpu)

if not args.debug:
    util.mkdirs(
    (path for key, path in opt['path'].items()
     if not key == 'experiments_root' and 'pretrain_model' not in key and 'resume' not in key))
util.setup_logger('base', opt['path']['log'], 'test_' + opt['name'], level=logging.INFO,
                  screen=True, tofile=not args.debug)
util.setup_logger('avg', opt['path']['log'], 'avg_' + opt['name'], level=logging.INFO,
                  screen=False, tofile=not args.debug)
logger = logging.getLogger('base')
logger.info(config.dict2str(opt))

#### Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt['datasets'].items()):
    test_set = LQGTDataset(dataset_opt)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1,
                                           pin_memory=True)
    logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
    test_loaders.append(test_loader)

# 模型
if opt['model_name'] == 'INNSR_model_1':
    from models.INNSR_model_1 import INNSRModel as M
elif opt['model_name'] == 'INNSR_model_2':
    from models.INNSR_model_2 import INNSRModel as M
elif opt['model_name'] == 'INNSR_model_0':
    from models.INNSR_model_0 import INNSRModel as M
else:
    raise NotImplementedError('Model [{:s}] is not defined.'.format(opt['model_name']))
model = M(opt)
for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt['name']
    logger.info('\nTesting [{:s}]...'.format(test_set_name))
    test_start_time = time.time()
    dataset_dir = osp.join(opt['path']['results_root'], test_set_name)
    util.mkdir(dataset_dir)

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['psnr_y'] = []
    test_results['ssim_y'] = []

    test_results['psnr_lr'] = []
    test_results['ssim_lr'] = []
    test_results['psnr_y_lr'] = []
    test_results['ssim_y_lr'] = []

    for data in test_loader:
        model.feed_data(data)
        img_path = data['GT_path'][0]
        img_name = osp.splitext(osp.basename(img_path))[0]

        model.test()
        visuals = model.get_current_visuals()

        lr_img = util.tensor2img(visuals['LR_ref'])
        sr_img = util.tensor2img(visuals['SR'])  # uint8
        srgt_img = util.tensor2img(visuals['GT'])  # uint8

        
        util.save_img(lr_img, osp.join(dataset_dir, img_name + '_LR.png'))
        util.save_img(sr_img, osp.join(dataset_dir, img_name + '_SR.png'))
        util.save_img(srgt_img, osp.join(dataset_dir, img_name + '_GT.png'))
        

        # calculate PSNR and SSIM
        gt_img = util.tensor2img(visuals['GT'])

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
        test_results['psnr'].append(psnr)
        test_results['ssim'].append(ssim)


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
            test_results['psnr_y'].append(psnr_y)
            test_results['ssim_y'].append(ssim_y)

            logger.info(
                    '{:20s} - PSNR: {:.6f} dB; SSIM: {:.6f}; PSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}.'.
                format(img_name, psnr, ssim, psnr_y, ssim_y))
            
        else:
            logger.info('{:20s} - PSNR: {:.6f} dB; SSIM: {:.6f}.'.format(img_name, psnr, ssim))

    # Average PSNR/SSIM results
    ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
    ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])

    logger.info(
            '----Average PSNR/SSIM results for {}----\n\tpsnr: {:.6f} db; ssim: {:.6f}. \n'.format(
            test_set_name, ave_psnr, ave_ssim))
    logger = logging.getLogger('avg')
    logger.info(
            '----Average PSNR/SSIM results for {}----\n\tpsnr: {:.6f} db; ssim: {:.6f}. \n'.format(
            test_set_name, ave_psnr, ave_ssim))
    logger = logging.getLogger('base')
            
    if test_results['psnr_y'] and test_results['ssim_y']:
        ave_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
        ave_ssim_y = sum(test_results['ssim_y']) / len(test_results['ssim_y'])

        logger.info(
            '----Y channel, average PSNR/SSIM----\n\tPSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}.'.
            format(ave_psnr_y, ave_ssim_y))
        logger = logging.getLogger('avg')
        logger.info(
            '----Y channel, average PSNR/SSIM----\n\tPSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}. \n'.
            format(ave_psnr_y, ave_ssim_y))
        logger = logging.getLogger('base')