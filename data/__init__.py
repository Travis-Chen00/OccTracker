from .nuscenes import build as build_nuscenes

def build_dataset(config, split):
    if config.DATASET == 'nuscenes':
        return build_nuscenes(config, split)
    return None