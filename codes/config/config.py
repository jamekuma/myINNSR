import os
import os.path as osp
import logging
import yaml

def parse(path, is_train=True):
    '''
    解析配置文件
    '''
    with open(path, mode='r') as f:
        opt = yaml.safe_load(f)
        
    # 在哪些gpu上运行
    gpu_list = ','.join(str(x) for x in opt['gpu_ids'])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    print('export CUDA_VISIBLE_DEVICES=' + gpu_list)
    
    # 维护路径信息
    opt['is_train'] = is_train
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir, osp.pardir)) # 项目的根目录
    opt['path']['root'] = root_path
    if is_train:
        experiments_root = osp.join(root_path, 'experiments', opt['name']) # 本次训练过程的记录目录
        opt['path']['experiments_root'] = experiments_root 
        opt['path']['models'] = osp.join(experiments_root, 'models')  # 中间模型的保存目录
        opt['path']['training_state'] = osp.join(experiments_root, 'training_state')  # 训练状态的保存目录
        opt['path']['log'] = experiments_root  # log目录
        opt['path']['val_images'] = osp.join(experiments_root, 'val_images')
    else:
        results_root = osp.join(opt['path']['root'], 'results', opt['name'])
        opt['path']['results_root'] = results_root
        opt['path']['log'] = results_root

    for phase, dataset in opt['datasets'].items():
        phase = phase.split('_')[0]
        dataset['phase'] = phase
        dataset['scale'] = opt['scale']
    
    return opt


def dict2str(opt, indent_l=1):
    '''dict to string for logger'''
    msg = ''
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_l * 2) + k + ':[\n'
            msg += dict2str(v, indent_l + 1)
            msg += ' ' * (indent_l * 2) + ']\n'
        else:
            msg += ' ' * (indent_l * 2) + k + ': ' + str(v) + '\n'
    return msg

class NoneDict(dict):
    def __missing__(self, key):
        return None

def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt
if __name__ == "__main__":
    opt = parse('./options/train_x4.yml')