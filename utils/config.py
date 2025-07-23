# This file is used to define the config
# File path "/config/model.yaml"

import torch
import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']

# ================================
#        Dataset config
# ================================
_C.DATA = CN()
_C.DATA.BATCH_SIZE = 1
_C.DATA.PATH = ""
_C.DATA.DATASET = "nuscenes"                         # May BDD100K AND SO ON LATER
_C.DATA.IMAGE_SIZE = 224
_C.DATA.INTERPOLATION = "bilinear"
_C.DATA.NUM_WORKERS = 8


# ===============================
#       Model config
# ===============================
_C.MODEL = CN()
_C.MODEL.BACKBONE = "resnet50"                      # Resnet50 or may be others later
_C.MODEL.PRETRAINED = ""                            # Pretrained model or not
_C.MODEL.NUM_CLASSES = 10                           # Num of class as the object query

_C.MODEL.DROP_RATE = 0.0
_C.MODEL.WINDOW_SIZE = 8                            # The size for window partition
_C.MODEL.DEVICE = "cuda"                            # Device
_C.MODEL.USE_CHECKPOINT = False

_C.MODEL.HIDDEN_DIM = 256
_C.MODEL.FFN_DIM = 2048
_C.MODEL.FEATURE_LEVEL = 4
_C.MODEL.NUM_HEADS = 8
_C.MODEL.NUM_ENC_POINTS = 4
_C.MODEL.NUM_DEC_POINTS = 4
_C.MODEL.NUM_ENC_LAYERS = 6
_C.MODEL.NUM_DEC_LAYERS = 6
_C.MODEL.ACTIVATION = "relu"
_C.MODEL.OCC_THRESHOLD = 0.5
_C.MODEL.MISS_PERIOD = 10
_C.MODEL.NUM_DET_QUERIES = 100                      # DETR proposal generation


# ===============================
#       Train config
# ===============================
_C.TRAIN = CN()
_C.TRAIN.EPOCH = 10                                 # Training epoch
_C.TRAIN.LR = 2.0e-4
_C.TRAIN.SEED = 40
_C.TRAIN.OPTIMIZER = "adamw"

def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    def _check_args(name):
        if hasattr(args, name) and eval(f'args.{name}'):
            return True
        return False

    # merge from specific arguments
    if _check_args('batch_size'):
        config.DATA.BATCH_SIZE = args.batch_size
    if _check_args('data_path'):
        config.DATA.DATA_PATH = args.data_path
    if _check_args('pretrained'):
        config.MODEL.PRETRAINED = args.pretrained
    if _check_args('use_checkpoint'):
        config.TRAIN.USE_CHECKPOINT = True
    if _check_args('optim'):
        config.TRAIN.OPTIMIZER = args.optim

    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config




