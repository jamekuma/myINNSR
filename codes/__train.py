import argparse
import torch
import config.config as config
import utils.utils as util
import logging 
from collections import OrderedDict
import random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from data.dataset import LQGTDataset
import math
import os.path as osp
import os




# 加载配置文件
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, help='Path to option YMAL file.')
parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                    help='job launcher')
parser.add_argument('--debug', help='debug mode, do not make any dict', action="store_true")
parser.add_argument('-gpu', type=str, help='gpu_id', default='0')

# parser.add_argument('--local_rank', type=int, default=0)
args = parser.parse_args()
opt = config.parse(args.opt, is_train=True)
opt['gpu_ids'] = [int(x) for x in args.gpu.split(',')]
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
print('export CUDA_VISIBLE_DEVICES=' + args.gpu)


# 建立目录、日志
if not opt['path'].get('resume_state', None) and not args.debug:  # 如果是恢复训练，则不重命名目录
    util.mkdir_and_rename(opt['path']['experiments_root'])
    util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                         and 'pretrain' not in key and 'resume' not in key))
util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO,
                    screen=True, tofile=not args.debug)
util.setup_logger('val', opt['path']['log'], 'val_' + opt['name'], level=logging.INFO,
                    screen=True, tofile=not args.debug)
util.setup_logger('test', opt['path']['log'], 'test_' + opt['name'], level=logging.INFO,
                    screen=True, tofile=not args.debug)
logger = logging.getLogger('base')
logger.info(config.dict2str(opt))  # 记录配置
tb_logger = SummaryWriter(log_dir='../tb_logger/' + opt['name'])

# 随机数种子
seed = random.randint(1, 10000)
logger.info('Random seed: {}'.format(seed))
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

opt = config.dict_to_nonedict(opt)  # 把配置变为NoneDict, 即不存在的键值为None
torch.backends.cudnn.benchmark = True

# 数据集
test_loaders = OrderedDict()
for phase, dataset_opt in sorted(opt['datasets'].items()):
    if phase == 'train':
        train_set = LQGTDataset(dataset_opt)
        train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
        total_iters = int(opt['train']['n_iter'])
        total_epochs = int(math.ceil(total_iters / train_size))
        # train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
        num_workers = dataset_opt['n_workers'] * len(opt['gpu_ids'])
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=dataset_opt['batch_size'], shuffle=True,
                                           num_workers=num_workers, drop_last=True,
                                           pin_memory=True)

        logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
        logger.info('Total epochs needed: {:d} for iters {:,d}'.format(total_epochs, total_iters))
    elif phase == 'val':
        val_set = LQGTDataset(dataset_opt)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1,
                                           pin_memory=True)
        logger.info('Number of val images in [{:s}]: {:d}'.format(dataset_opt['name'], len(val_set)))
    elif phase.startswith('test'):
        test_set = LQGTDataset(dataset_opt)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1,
                                           pin_memory=True)
        # test_loaders.append(test_loader)
        test_loaders[dataset_opt['name']] = test_loader
        logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
    else:
        raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))



# 从指定目录中加载训练状态（用于恢复上次的训练）
if opt['path'].get('resume_state', None):
    device_id = torch.cuda.current_device()
    resume_state = torch.load(opt['path']['resume_state'],
                              map_location=lambda storage, loc: storage.cuda(device_id))
    resume_iter = resume_state['iter']
    # option.check_resume(opt, resume_state['iter'])  # check resume options
    if opt['path'].get('pretrain_INN', None) is not None or opt['path'].get(
                'pretrain_GapNN', None) is not None:
        logger.warning('pretrain_model path will be ignored when resuming training.')

    opt['path']['pretrain_INN'] = osp.join(opt['path']['models'], '{}_INN.pth'.format(resume_iter))
    opt['path']['pretrain_GapNN'] = osp.join(opt['path']['models'], '{}_GapNN.pth'.format(resume_iter))
    logger.info('Set [pretrain_INN] to ' + opt['path']['pretrain_INN'])
    logger.info('Set [pretrain_GapNN] to ' + opt['path']['pretrain_GapNN'])

    logger.info('Resuming training from epoch: {}, iter: {}.'.format(
        resume_state['epoch'], resume_state['iter']))
    start_epoch = resume_state['epoch']
    current_step = resume_state['iter']
else:
    current_step = 0
    start_epoch = 0

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

if opt['path'].get('resume_state', None):
    model.resume_training(resume_state)  # handle optimizers and schedulers

#### 开始训练
logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
for epoch in range(start_epoch, total_epochs + 1):
    for train_data in train_loader:
        current_step += 1
        if current_step > total_iters:
            break
        model.feed_data(train_data)
        model.optimize_parameters(current_step)
        model.update_learning_rate(current_step, warmup_iter=opt['train']['warmup_iter'])

        # log
        if current_step % opt['logger']['print_freq'] == 0:
            logs = model.get_current_log()
            message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(
                epoch, current_step, model.get_current_learning_rate())
            for k, v in logs.items():
                message += '{:s}: {:.4e} '.format(k, v)
                # tensorboard logger
                if opt['use_tb_logger']:
                    tb_logger.add_scalar(k, v, current_step)       
            logger.info(message)

        # validation
        if current_step % opt['train']['val_freq'] == 0:
            avg_psnr_SR = 0.0
            cnt = 0
            for val_data in val_loader:
                cnt += 1
                model.feed_data(val_data)
                model.test()
                crop_size = opt['scale']
                visuals = model.get_current_visuals()
                sr_img = util.tensor2img(visuals['SR'])  # uint8
                gt_img = util.tensor2img(visuals['GT'])  # uint8
                lr_forw_img = util.tensor2img(visuals['LR_forw'])
                # gtl_img = util.tensor2img(visuals['LR_ref'])

                gt_img = gt_img / 255.
                sr_img = sr_img / 255.
                lr_forw_img = lr_forw_img / 255.

                cropped_sr_img = sr_img[crop_size:-crop_size, crop_size:-crop_size, :]
                cropped_gt_img = gt_img[crop_size:-crop_size, crop_size:-crop_size, :]
                cropped_lr_forw_img = lr_forw_img[1:-1, 1:-1, :]
                
                avg_psnr_SR += util.calculate_psnr(cropped_sr_img * 255, cropped_gt_img * 255)
            avg_psnr_SR = avg_psnr_SR / cnt
            # log
            logger.info('# Validation # PSNR_SR: {:.4e}.'.format(avg_psnr_SR))
            logger_val = logging.getLogger('val')  # validation logger
            logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr_sr: {:.4e}.'.format(epoch, current_step, avg_psnr_SR))
            # tensorboard logger
            if opt['use_tb_logger']:
                tb_logger.add_scalar('psnr_sr', avg_psnr_SR, current_step)
        
        # 保存模型及训练数据
        if current_step % opt['logger']['save_checkpoint_freq'] == 0:
            logger.info('Saving models and training states.')
            model.save(current_step)
            model.save_training_state(epoch, current_step)

        # test
        if current_step % opt['train']['test_freq'] == 0:
            log_msg = '<epoch:{:3d}, iter:{:8,d}> psnr_sr: {:.4e}. Y_channel PSNR/SSIM: \n'
            for name, test_loader in test_loaders.items():
                avg_psnr_y_SR = 0.0
                avg_ssim_y_SR = 0.0
                cnt = 0
                log_msg += name + ': '
                for test_data in test_loader:
                    cnt += 1
                    model.feed_data(test_data)
                    model.test()
                    crop_size = opt['scale']
                    visuals = model.get_current_visuals()

                    sr_img = util.tensor2img(visuals['SR'])  # uint8
                    gt_img = util.tensor2img(visuals['GT'])  # uint8

                    gt_img = gt_img / 255.
                    sr_img = sr_img / 255.

                    sr_img_y = util.bgr2ycbcr(sr_img, only_y=True)
                    gt_img_y = util.bgr2ycbcr(gt_img, only_y=True)

                    cropped_sr_img_y = sr_img_y[crop_size:-crop_size, crop_size:-crop_size]
                    cropped_gt_img_y = gt_img_y[crop_size:-crop_size, crop_size:-crop_size]
                    
                    avg_psnr_y_SR += util.calculate_psnr(cropped_sr_img_y * 255, cropped_gt_img_y * 255)
                    avg_ssim_y_SR += util.calculate_ssim(cropped_sr_img_y * 255, cropped_gt_img_y * 255)
                avg_psnr_y_SR = avg_psnr_y_SR / cnt
                avg_ssim_y_SR = avg_ssim_y_SR / cnt
                log_msg += '{:.6f}db/{:.6f} \n'.format(avg_psnr_y_SR, avg_ssim_y_SR)
            logger_test = logging.getLogger('test')
            logger_test.info(log_msg)
            

        
