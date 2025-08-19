from .tracker import build_tracker
import torch

def build_model(args):
    model = build_tracker(args)
    model.to(device=torch.device(args.DEVICE))
    return model