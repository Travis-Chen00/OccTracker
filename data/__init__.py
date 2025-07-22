from .nuscenes import build as build_nuscenes

def build_dataset(image_set, args):
    if args.dataset == 'nuscenes':
        return build_nuscenes(image_set, args)