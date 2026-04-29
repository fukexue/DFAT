import os
import os.path as osp
import argparse
import yaml
from easydict import EasyDict as edict
from geotransformer.utils.common import ensure_dir


# new version for argparse and config file, disable for make cfg
def default_parse_args(stage='train'):
    parser = argparse.ArgumentParser()
    # common
    parser.add_argument('--cfg', type=str, default='', help='config file')
    parser.add_argument('--note_name', type=str, default='Debug', help='note for output')
    # different stage have different para
    if stage == 'train':
        parser.add_argument('--resume', action='store_true', help='resume training')
        parser.add_argument('--snapshot', default=None, help='load from snapshot')
        parser.add_argument('--epoch', type=int, default=None, help='load epoch')
        parser.add_argument('--log_steps', type=int, default=10, help='logging steps')
        parser.add_argument('--local_rank', type=int, default=-1, help='local rank for ddp')
    elif stage == 'test':
        parser.add_argument('--snapshot', default=None, help='load from snapshot')
        parser.add_argument('--test_iter', type=int, default=None, help='test iteration')
    elif stage == 'eval':
        parser.add_argument('--method', choices=['lgr', 'ransac', 'svd', 'mmc'], required=True, help='registration method')
        parser.add_argument('--num_corr', type=int, default=None, help='number of correspondences for registration')
        parser.add_argument('--verbose', action='store_true', help='verbose mode')
    else:
        RuntimeError

    args = parser.parse_args()

    # load Yaml
    config_path = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.realpath(__file__)))), 'config', args.cfg)
    if stage == 'test' or stage == 'eval':
        if osp.exists(osp.join(osp.dirname(osp.dirname(osp.dirname(osp.realpath(__file__)))), 'output', 'Kitti_'+args.note_name, 'config.yaml')):
            config_path = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.realpath(__file__)))), 'output', 'Kitti_'+args.note_name, 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)

    # make dir
    config['working_dir'] = osp.dirname(osp.realpath(__file__))
    config['root_dir'] = osp.dirname(osp.dirname(config['working_dir']))
    config['exp_name'] = osp.basename(config['working_dir'])
    config['output_dir'] = osp.join(config['root_dir'], 'output', config['exp_name']+'_'+args.note_name)
    config['snapshot_dir'] = osp.join(config['output_dir'], 'snapshots')
    config['log_dir'] = osp.join(config['output_dir'], 'logs_3DLoMatch' if hasattr(args, 'benchmark') and args.benchmark=='3DLoMatch' else 'logs')
    config['event_dir'] = osp.join(config['output_dir'], 'events')
    config['feature_dir'] = osp.join(config['output_dir'], 'features_3DLoMatch' if hasattr(args, 'benchmark') and args.benchmark=='3DLoMatch' else 'features')

    ensure_dir(config['output_dir'])
    ensure_dir(config['snapshot_dir'])
    ensure_dir(config['log_dir'])
    ensure_dir(config['event_dir'])
    ensure_dir(config['feature_dir'])

    # Supplementary parameters: yaml file cannot input
    # data path
    config['data'] = {}
    config['data']['dataset_root'] = osp.join(config['root_dir'], 'data', 'Kitti')
    # math compute
    config['backbone']['init_radius'] = config['backbone']['base_radius'] * config['backbone']['init_voxel_size']
    config['backbone']['init_sigma'] = config['backbone']['base_sigma'] * config['backbone']['init_voxel_size']
    # None var
    config['test']['point_limit'] = None
    config['fine_matching']['correspondence_limit'] = None

    config = edict(config)

    # save yaml file only for training
    if stage == 'train':
        os.system(f'cp -r {config_path} {config.output_dir}/config.yaml')

    return args, config
