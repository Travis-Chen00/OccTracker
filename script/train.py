# Data loader
# Model Train (Frame by frame as input)
# logger
import os
from log.logger import Logger
import argparse
from data import build_dataset
from model import build_model
import torch
from utils.utils import set_seed
from utils.config import get_config


def parse_options():
    parser = argparse.ArgumentParser('OccTracker Training', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--optim', type=str,
                        help='overwrite optimizer if provided, can be adamw/sgd/fused_adam/fused_lamb.')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def train(args):
    train_logger = Logger(logdir=os.path.join(args.OUTPUT_DIR, "train"), only_main=True)
    train_logger.show(head="Args: ", log=args)
    train_logger.write(log=args, filename="config.yaml", mode="w")

    set_seed(args.SEED)

    model = build_model(args)

if __name__ == '__main__':
    args, config = parse_options()

