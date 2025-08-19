import os
import yaml
import torch
import random
import torch.distributed
import torch.backends.cudnn
import numpy as np

from .misc import is_main_process

def get_model(model):
    return model if is_distributed() is False else model.module

def is_distributed():
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return False
    return True

def distributed_rank():
    if not is_distributed():
        return 0
    else:
        return torch.distributed.get_rank()

def distributed_world_size():
    if is_distributed():
        return torch.distributed.get_world_size()
    else:
        return 1

def set_seed(seed: int):
    seed = seed + distributed_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # If you don't want to wait until the universe is silent, do not use this below code :)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    return

def save_checkpoint(model, path, states,
                    optimizer, scheduler):
    model = get_model(model)
    if is_main_process():
        save_state = {
            "model": model.state_dict(),
            "optimizer": None if optimizer is None else optimizer.state_dict(),
            "scheduler": None if scheduler is None else scheduler.state_dict(),
            'states': states
        }
        torch.save(save_state, path)
    else:
        pass
    return


def load_pretrained_model(model, pretrained_path):
    # May be rewrite to fit the model
    if not is_main_process():
        return model
    pretrained_checkpoint = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    pretrained_state_dict = pretrained_checkpoint["model"]
    model_state_dict = model.state_dict()

    pretrained_keys = list(pretrained_state_dict.keys())
    for k in pretrained_keys:
        if k in model_state_dict:
            if model_state_dict[k].shape != pretrained_state_dict[k].shape:
                if "class_embed" in k:
                    if model_state_dict[k].shape[0] == 1:
                        pretrained_state_dict[k] = pretrained_state_dict[k][1:2]
                    elif model_state_dict[k].shape[0] == 2:
                        pretrained_state_dict[k] = pretrained_state_dict[k][1:3]
                    elif model_state_dict[k].shape[0] == 3:
                        pretrained_state_dict[k] = pretrained_state_dict[k][1:4]
                    elif model_state_dict[k].shape[0] == 8:     # BDD100K
                        pretrained_state_dict[k] = model_state_dict[k]
                        # We directly do not use the pretrained class embed for BDD100K
                    else:
                        raise NotImplementedError('invalid shape: {}'.format(model_state_dict[k].shape))
                else:
                    print(f"Parameter {k} has shape{pretrained_state_dict[k].shape} in pretrained model, "
                          f"but get shape{model_state_dict[k].shape} in current model.")
        elif "query_embed" in k:
            if pretrained_state_dict[k].shape == model_state_dict["det_query_embed"].shape:
                pretrained_state_dict["det_query_embed"] = pretrained_state_dict[k].clone()
            else:
                print(f"Det Query shape is not equal. Check if you turn on 'USE_DAB'.")
                pretrained_state_dict["det_query_embed"] = model_state_dict["det_query_embed"]
            del pretrained_state_dict[k]
        elif "tgt_embed" in k:  # for DAB
            if pretrained_state_dict[k].shape == model_state_dict["det_query_embed"].shape:
                pretrained_state_dict["det_query_embed"] = pretrained_state_dict[k].clone()
            else:
                pretrained_state_dict["det_query_embed"] = model_state_dict["det_query_embed"]
            del pretrained_state_dict[k]
        elif "refpoint_embed" in k:
            if pretrained_state_dict[k].shape == model_state_dict["det_anchor"].shape:
                pretrained_state_dict["det_anchor"] = pretrained_state_dict[k].clone()
            else:
                pretrained_state_dict["det_anchor"] = model_state_dict["det_anchor"]
                print(f"Pretrain model's query num is {pretrained_state_dict[k].shape[0]}, "
                      f"current model's query num is {model_state_dict['det_anchor'].shape[0]}, "
                      f"do not load these parameters.")
            del pretrained_state_dict[k]
        elif "backbone" in k:
            new_k = k[15:]
            new_k = "backbone.backbone.backbone" + new_k
            pretrained_state_dict[new_k] = pretrained_state_dict[k].clone()
            del pretrained_state_dict[k]
        elif "input_proj" in k:
            new_k = k[10:]
            new_k = "feature_projs" + new_k
            pretrained_state_dict[new_k] = pretrained_state_dict[k].clone()
            del pretrained_state_dict[k]
        else:
            pass

    not_in_model = 0
    for k in pretrained_state_dict:
        if k not in model_state_dict:
            not_in_model += 1
            # if show_details:
            #     print(f"Parameter {k} in the pretrained model but not in the current model.")

    not_in_pretrained = 0
    for k in model_state_dict:
        if k not in pretrained_state_dict:
            pretrained_state_dict[k] = model_state_dict[k]
            not_in_pretrained += 1
            # if show_details:
            #     print(f"There is a new parameter {k} in the current model, but not in the pretrained model.")

    model.load_state_dict(state_dict=pretrained_state_dict, strict=False)
    print(f"Pretrained model is loaded, there are {not_in_model} parameters droped "
          f"and {not_in_pretrained} parameters unloaded, set 'show details' True to see more details.")

    return model
