import torch
import os
import numpy as np
from collections import defaultdict
from PIL import Image

from nuscenes.nuscenes import NuScenes as NuScenesAPI
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points, box_in_image
from nuscenes.utils.splits import create_splits_scenes
from pyquaternion import Quaternion

import data.transforms as T


class NuScenesTemporal:
    def __init__(self, args, transform=None, split='train', temporal=True, num_frames=2):
        """
        args: dict 配置参数
        transform: 数据增强 pipeline
        split: 'train' / 'val' / 'test'
        temporal: 是否加载时序数据（True 时返回 num_frames 帧）
        num_frames: 时序长度（包含当前帧）
        """
        self.args = args
        self.transform = transform
        self.split = split
        self.temporal = temporal
        self.num_frames = num_frames

        # 类别映射
        self.category_mapping = {
            'vehicle.car': 0, 'vehicle.truck': 1,
            'vehicle.bus.rigid': 2, 'vehicle.bus.bendy': 2,
            'vehicle.trailer': 3,
            'vehicle.motorcycle': 4,
            'vehicle.bicycle': 5,
            'human.pedestrian.adult': 6,
            'human.pedestrian.child': 6,
            'human.pedestrian.construction_worker': 6,
            'human.pedestrian.police_officer': 6,
            'animal': 7
        }

        self.filter_categories = args.get("FILTER_CATEGORIES", None)
        self.min_visibility = args.get("MIN_VISIBILITY", 0)
        self.camera_types = args.get("CAMERA_TYPES", ['CAM_FRONT'])

        self.samples = []             # 所有采样帧
        self.scene_samples = defaultdict(list)  # 场景 -> 样本列表

        self._validate_args(args)
        self.nusc = NuScenesAPI(
            version=args.get("VERSION", "v1.0-trainval"),
            dataroot=args["DATA_ROOT"],
            verbose=False
        )
        self._load_samples()

    def _validate_args(self, args):
        assert "DATA_ROOT" in args, "需要设置 DATA_ROOT"
        assert os.path.exists(args["DATA_ROOT"]), f"{args['DATA_ROOT']} 不存在"

    def _load_samples(self):
        """原始模式：从 nuScenes API 按场景加载帧信息"""
        print(f"[NuScenesTemporal] 加载 {self.split} 数据...")
        if self.split in ['train', 'val']:
            scene_splits = create_splits_scenes()
            scenes = scene_splits[self.split]
        else:
            scenes = [scene['name'] for scene in self.nusc.scene]

        sample_count = 0
        for scene_name in scenes:
            scene = next((s for s in self.nusc.scene if s['name'] == scene_name), None)
            if scene is None:
                continue
            sample_token = scene['first_sample_token']
            scene_sample_idx = 0
            while sample_token:
                sample = self.nusc.get('sample', sample_token)
                for cam_type in self.camera_types:
                    if cam_type in sample['data']:
                        sample_info = {
                            'sample_token': sample_token,
                            'cam_token': sample['data'][cam_type],
                            'cam_type': cam_type,
                            'scene_name': scene_name,
                            'scene_idx': scene_sample_idx,
                            'global_idx': len(self.samples)
                        }
                        self.samples.append(sample_info)
                        self.scene_samples[scene_name].append(sample_info)
                        sample_count += 1
                sample_token = sample['next']
                scene_sample_idx += 1
        print(f"[NuScenesTemporal] 加载完成: {sample_count} 帧 ({len(scenes)} 场景)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """加载单条数据（单帧 or 时序）"""
        if self.temporal:
            # 获取当前帧及之前 num_frames-1 帧
            curr_info = self.samples[idx]
            frames = self._get_prev_frames(curr_info, self.num_frames)
            imgs, infos = self.get_multi_frames(frames)
        else:
            img, info = self.get_single_frame(self.samples[idx])
            imgs, infos = [img], [info]

        # 数据增强
        if self.transform:
            imgs, infos = self.transform(imgs, infos)

        # 每帧 target
        targets = []
        for info in infos:
            targets.append({
                "boxes": info["boxes"],
                "labels": info["labels"],
                "obj_ids": info["ids"],
                "area": info["areas"]
            })

        image_sizes = [torch.tensor([img.shape[1], img.shape[2]], dtype=torch.long) for img in imgs]
        image_size = image_sizes[0] if len(image_sizes) == 1 else image_sizes

        return {
            "imgs": imgs,            # list[Tensor], 每个是 CxHxW
            "infos": infos,          # list[dict]，包含 boxes、labels 等
            "image_size": image_size,
            "gt_instances": targets
        }

    def _get_prev_frames(self, curr_info, num_frames):
        """在同一场景内向前取 num_frames 帧，不够就重复第一帧"""
        scene_name = curr_info['scene_name']
        scene_samples = self.scene_samples[scene_name]
        curr_idx = curr_info['scene_idx']

        # 从当前帧往前取
        start_idx = max(0, curr_idx - (num_frames - 1))
        frames = scene_samples[start_idx:curr_idx + 1]

        # 如果不够帧数，前面补齐
        while len(frames) < num_frames:
            frames.insert(0, frames[0])

        return frames

    def get_single_frame(self, sample_info):
        sample = self.nusc.get('sample', sample_info['sample_token'])
        cam_data = self.nusc.get('sample_data', sample_info['cam_token'])
        img_path = os.path.join(self.nusc.dataroot, cam_data['filename'])
        img = Image.open(img_path).convert('RGB')

        calibrated_sensor = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        ego_pose = self.nusc.get('ego_pose', cam_data['ego_pose_token'])

        boxes_2d, ids, labels, visibilities = self._get_2d_boxes(
            sample, sample_info['cam_type'], calibrated_sensor, ego_pose, img.size
        )

        info = {
            'boxes': torch.tensor(boxes_2d if boxes_2d else [[0.5, 0.5, 0.01, 0.01]], dtype=torch.float32),
            'ids': torch.tensor(ids if ids else [0], dtype=torch.long),
            'labels': torch.tensor(labels if labels else [0], dtype=torch.long),
            'areas': torch.tensor([b[2]*b[3] for b in boxes_2d] if boxes_2d else [0.0001], dtype=torch.float32),
            'visibilities': torch.tensor(visibilities if visibilities else [1], dtype=torch.long),
            'frame_idx': torch.tensor(sample_info.get('scene_idx', 0), dtype=torch.long),
            'timestamp': sample['timestamp'],
            'sample_token': sample_info['sample_token'],
            'cam_type': sample_info['cam_type']
        }
        return img, info

    def _get_2d_boxes(self, sample, cam_type, calibrated_sensor, ego_pose, img_size):
        boxes_2d, ids, labels, visibilities = [], [], [], []
        camera_intrinsic = np.array(calibrated_sensor['camera_intrinsic'])
        for ann_token in sample['anns']:
            ann = self.nusc.get('sample_annotation', ann_token)
            category = ann['category_name']
            if self.filter_categories and not any(cat in category for cat in self.filter_categories):
                continue
            label = self._get_label_from_category(category)
            if label is None:
                continue
            box_3d = Box(ann['translation'], ann['size'], Quaternion(ann['rotation']))
            box_3d.translate(-np.array(ego_pose['translation']))
            box_3d.rotate(Quaternion(ego_pose['rotation']).inverse)
            box_3d.translate(-np.array(calibrated_sensor['translation']))
            box_3d.rotate(Quaternion(calibrated_sensor['rotation']).inverse)
            if not box_in_image(box_3d, camera_intrinsic, img_size, vis_level=self.min_visibility):
                continue
            visibility = self.nusc.get('visibility', ann['visibility_token'])
            vis_level = int(visibility['token'])
            if vis_level < self.min_visibility:
                continue
            corners_2d = view_points(box_3d.corners(), camera_intrinsic, normalize=True)
            x_min, y_min = corners_2d[:2].min(axis=1)
            x_max, y_max = corners_2d[:2].max(axis=1)
            x, y = x_min / img_size[0], y_min / img_size[1]
            w, h = (x_max - x_min) / img_size[0], (y_max - y_min) / img_size[1]
            if w > 0 and h > 0:
                boxes_2d.append([x + w/2, y + h/2, w, h])
                ids.append(int(ann['instance_token'][:8], 16) % 10000)
                labels.append(label)
                visibilities.append(vis_level)
        return boxes_2d, ids, labels, visibilities

    def _get_label_from_category(self, category):
        for prefix, label in self.category_mapping.items():
            if category.startswith(prefix):
                return label
        return None

    def get_multi_frames(self, frame_samples):
        frames_data = [self.get_single_frame(s) for s in frame_samples]
        imgs, infos = zip(*frames_data)
        return list(imgs), list(infos)


# --------------------------
#  数据增强
# --------------------------
def transforms_for_train():
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    return T.MultiCompose([
        T.MultiRandomHorizontalFlip(),
        T.MultiRandomSelect(
            T.MultiRandomResize(sizes=scales, max_size=1333),
            T.MultiCompose([
                T.MultiRandomResize([400, 500, 600]),
                T.MultiRandomCrop(min_size=384, max_size=600, overflow_bbox=True),
                T.MultiRandomResize(sizes=scales, max_size=1333)
            ])
        ),
        T.MultiHSV(),
        T.MultiCompose([
            T.MultiToTensor(),
            T.MultiNormalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        ])
    ])


def transforms_for_val():
    return T.MultiCompose([
        T.MultiRandomResize([800], max_size=1333),
        T.MultiCompose([
            T.MultiToTensor(),
            T.MultiNormalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        ])
    ])


def build(args, split='train', temporal=True, num_frames=2):
    transform = transforms_for_train() if split == 'train' else transforms_for_val()
    return NuScenesTemporal(args=args, transform=transform,
                            split=split, temporal=temporal,
                            num_frames=num_frames)


# --------------------------
#  安全的 collate_fn
# --------------------------
def nuscenes_collate_fn(batch):
    """
    防止 default_collate 在处理 list[Tensor] 形状不一致时报错。
    会对 imgs/infos/gt_instances 保持 list，不进行 stack。
    """
    collated_batch = {}
    for key in batch[0]:
        elems = [d[key] for d in batch]
        if key in ["imgs", "infos", "gt_instances"]:
            collated_batch[key] = elems  # 保持原 list
        else:
            try:
                collated_batch[key] = torch.utils.data._utils.collate.default_collate(elems)
            except:
                collated_batch[key] = elems
    return collated_batch
