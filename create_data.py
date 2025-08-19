import argparse
import nuscene_converter as nuscenes_converter
from os import path as osp

def nuscenes_data_prep(root_path,
                       info_prefix,
                       version,
                       dataset_name,
                       out_dir,
                       max_sweeps=10):
    """Prepare data related to nuScenes dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        dataset_name (str): The dataset class name.
        out_dir (str): Output directory of the groundtruth database info.
        max_sweeps (int): Number of input consecutive frames. Default: 10
    """
    nuscenes_converter.create_nuscenes_infos(
        root_path, out_dir, info_prefix, version=version, max_sweeps=max_sweeps)

    if version == 'v1.0-test':
        info_test_path = osp.join(
            out_dir, f'{info_prefix}_infos_temporal_test.pkl')
        nuscenes_converter.export_2d_annotation(
            root_path, info_test_path, version=version)
    else:
        info_train_path = osp.join(
            out_dir, f'{info_prefix}_infos_temporal_train.pkl')
        info_val_path = osp.join(
            out_dir, f'{info_prefix}_infos_temporal_val.pkl')
        nuscenes_converter.export_2d_annotation(
            root_path, info_train_path, version=version)
        nuscenes_converter.export_2d_annotation(
            root_path, info_val_path, version=version)

def main():
    parser = argparse.ArgumentParser(description='NuScenes Video Sequence Converter')
    parser.add_argument('--dataset', type=str, default='nuscenes',
                        help='specify the dataset')
    parser.add_argument('--root-path', type=str, default='/home/boyu/Desktop/OccTracker/dataset/v1.0-mini',
                        help='specify the root path of dataset')
    parser.add_argument('--version', type=str, default='v1.0-mini',
                        help='specify the dataset version')
    parser.add_argument('--out_dir', type=str, default='/home/boyu/Desktop/OccTracker/dataset/v1.0-mini',
                        help='output directory')
    parser.add_argument('--max-sweeps', type=int, default=10,
                        help='max number of sweeps')
    parser.add_argument('--extra_tag', type=str, default='nuscenes',
                        help='camera type to use')
    parser.add_argument('--sequence-length', type=int, default=5,
                        help='length of video sequences')

    args = parser.parse_args()
    if args.dataset == 'nuscenes' and args.version != 'v1.0-mini':
        train_version = f'{args.version}-trainval'
        nuscenes_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=train_version,
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps)
        test_version = f'{args.version}-test'
        nuscenes_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=test_version,
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps)
    elif args.dataset == 'nuscenes' and args.version == 'v1.0-mini':
        train_version = f'{args.version}'
        nuscenes_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=train_version,
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps)

if __name__ == '__main__':
    main()
