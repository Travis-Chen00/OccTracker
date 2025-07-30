# Data loader
# Model Train (Frame by frame as input)
# logger
import os
import argparse
import time

from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from torch.testing._internal.common_quantization import AverageMeter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader

from data import build_dataset
from model import build_model
import torch
from utils.utils import set_seed, load_pretrained_model, distributed_rank, is_distributed, save_checkpoint, get_model
from utils.config import get_config
from logger import Logger
from model.criterion import build_criterion
from model.structures.tracks import Tracks

print("train")

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


def train(config):
    train_logger = Logger(logdir=os.path.join(config.OUTPUT_DIR, "train"), only_main=True)
    train_logger.show(head="Config: ", log=config)
    train_logger.write(head=" ", log=config, filename="config.yaml", mode="w")

    set_seed(config.SEED)

    model = build_model(config)

    # Load Pretrained Model
    if config.PRETRAINED_MODEL is not None:
        model = load_pretrained_model(model, config.PRETRAINED_MODEL)

    # Data process
    dataset_train = build_dataset(config=config, split="train")
    # sampler_train = build_sampler(dataset=dataset_train, shuffle=True)
    # dataloader_train = build_dataloader(dataset=dataset_train, sampler=sampler_train,
    #                                     batch_size=config["BATCH_SIZE"], num_workers=config["NUM_WORKERS"])

    # Criterion
    criterion = build_criterion(args=config)
    criterion.set_device(torch.device("cuda", distributed_rank()))

    # Optimizer
    param_groups, lr_names = get_param_groups(config=config, model=model)
    optimizer = AdamW(params=param_groups, lr=config.LR)
    # Scheduler
    if config["LR_SCHEDULER"] == "MultiStep":
        scheduler = MultiStepLR(
            optimizer,
            milestones=config["LR_DROP_MILESTONES"],
            gamma=config["LR_DROP_RATE"]
        )
    elif config["LR_SCHEDULER"] == "Cosine":
        scheduler = CosineAnnealingLR(
            optimizer=optimizer,
            T_max=config["EPOCHS"]
        )
    else:
        raise ValueError(f"Do not support lr scheduler '{config['LR_SCHEDULER']}'")

    # Training states
    train_states = {
        "start_epoch": 0,
        "global_iters": 0
    }

    # Set start epoch
    start_epoch = train_states["start_epoch"] if train_states["start_epoch"] > 0 else 0


    if is_distributed():
        model = DDP(module=model, device_ids=[distributed_rank()], find_unused_parameters=False)

    for epoch in range(start_epoch, config.EPOCH):
        # if is_distributed():
        #     sampler_train.set_epoch(epoch)
        dataset_train.set_epoch(epoch)
        sampler_train = RandomSampler(dataset_train) if config.SHUFFLE is True else SequentialSampler(dataset_train)
        dataloader_train = DataLoader(dataset=dataset_train, sampler=sampler_train,
                                            batch_size=config.DATA.BATCH_SIZE, num_workers=config.NUM_WORKERS,)

        lrs = [optimizer.param_groups[_]["lr"] for _ in range(len(optimizer.param_groups))]
        lr_info = [{name: lr} for name, lr in zip(lr_names, lrs)]
        train_logger.show(head=f"[Epoch {epoch}] lr={lr_info}")
        train_logger.write(head=f"[Epoch {epoch}] lr={lr_info}")

        train_one_epoch(
            epoch=epoch,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            dataloader=dataloader_train,
            # metric_log=train_metric_log,
            logger=train_logger,
            train_states=train_states,
        )
        scheduler.step()
        train_states["start_epoch"] += 1

        if config["EPOCHS"] < 100 or (epoch + 1) % 5 == 0:
            save_checkpoint(
                model=model,
                path=os.path.join(config["OUTPUTS_DIR"], f"checkpoint_{epoch}.pth"),
                states=train_states,
                optimizer=optimizer,
                scheduler=scheduler
            )

    return

def train_one_epoch(epoch, model, criterion, optimizer, dataloader,
                    train_states, logger):
    """
        epoch: epoch number
        model: model
        criterion: criterion
        optimizer: optimizer
        dataloader: dataloader
        logger: logger

        ===============================================
        return: logging information
    """
    model.train()
    optimizer.zero_grad()
    device = next(get_model(model).parameters()).device

    dataloader_len = len(dataloader)
    epoch_start_timestamp = time.time()
    for i, batch in enumerate(dataloader):
        iter_start_timestamp = time.time()
        tracks = Tracks._init_tracks(image_size=batch["image_size"],
                                     hidden_dim=get_model(model).hidden_dim,
                                     num_queries=get_model(model).num_queries,
                                     mem_bank_len=get_model(model).mem_bank_len,
                                     device=device)
        criterion.initialize_for_single_clip(batch['gt_instances'])

def get_param_groups(config, model):
    """
    用于针对不同部分的参数使用不同的 lr 等设置
    Args:
        config: 实验的配置信息
        model: 需要训练的模型

    Returns:
        params_group: a list of params groups.
        lr_names: a list of params groups' lr name, like "lr_backbone".
    """
    def match_keywords(name, keywords):
        matched = False
        for keyword in keywords:
            if keyword in name:
                matched = True
                break
        return matched
    # keywords
    backbone_keywords = ["backbone.backbone"]
    points_keywords = ["reference_points", "sampling_offsets"]  # 在 transformer 中用于选取参考点和采样点的网络参数关键字
    query_updater_keywords = ["query_updater"]
    param_groups = [
        {   # backbone 学习率设置
            "params": [p for n, p in model.named_parameters() if match_keywords(n, backbone_keywords) and p.requires_grad],
            "lr": config.LR_BACKBONE,
        },
        # {
        #     "params": [p for n, p in model.named_parameters() if match_keywords(n, points_keywords)
        #                and p.requires_grad],
        #     "lr": config.LR_POINT,
        # },
        # {
        #     "params": [p for n, p in model.named_parameters() if match_keywords(n, query_updater_keywords)
        #                and p.requires_grad],
        #     "lr": config.LR
        # },
        {
            "params": [p for n, p in model.named_parameters() if not match_keywords(n, backbone_keywords)
                       and not match_keywords(n, points_keywords)
                       and not match_keywords(n, query_updater_keywords)
                       and p.requires_grad],
            "lr": config.LR
        }
    ]
    return param_groups, ["lr_backbone", "lr_points", "lr"]

if __name__ == '__main__':
    args, config = parse_options()
