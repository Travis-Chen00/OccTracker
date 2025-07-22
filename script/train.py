# Data loader
# Model Train (Frame by frame as input)
# logger
import os
from log.logger import Logger

from data import build_dataset
from model import build_model
import torch
from utils.utils import set_seed

def train(args):
    train_logger = Logger(logdir=os.path.join(args.OUTPUT_DIR, "train"), only_main=True)
    train_logger.show(head="Args: ", log=args)
    train_logger.write(log=args, filename="config.yaml", mode="w")

    set_seed(args.SEED)

    model = build_model(args)


