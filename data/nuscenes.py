import torch
import os
from collections import defaultdict
from pathlib import Path
from PIL import Image
from math import floor
from random import randint
import data.transforms as T


class NuScenes():
    def __init__(self, args, transform, split):
        self.args = args
        self.transform = transform
        assert split == "train", f"Split {split} is not supported!"

        self.images_dir = os.path.join(args["DATA_ROOT"], "nuScenes", "images/track/train/")
        self.gts_dir = os.path.join(args["DATA_ROOT"], "nuScenes", "filter_labels/track/train/")
        assert os.path.exists(self.images_dir), f"Dir {self.images_dir} does not exist."
        assert os.path.exists(self.gts_dir), f"Dir {self.gts_dir} does not exist."

        # 验证采样参数
        self._validate_sample_params(args)

        # 采样的逻辑：
        self.sample_steps = args["SAMPLE_STEPS"]
        self.sample_intervals = args["SAMPLE_INTERVALS"]
        self.sample_modes = args["SAMPLE_MODES"]
        self.sample_lengths = args["SAMPLE_LENGTHS"]

        # 当前的采样策略，随着 epoch 的迭代，如下的内容应该会发生变化
        self.sample_stage = None
        self.sample_begin_frames = None
        self.sample_length = None
        self.sample_mode = None
        self.sample_interval = None
        self.sample_vid_tmax = None

        self.gts = defaultdict(lambda: defaultdict(list))
        self._load_annotations()
        self.set_epoch(0)  # init for each epoch

    def _validate_sample_params(self, args):
        """验证采样参数的有效性"""
        required_keys = ["SAMPLE_STEPS", "SAMPLE_INTERVALS", "SAMPLE_MODES", "SAMPLE_LENGTHS"]
        for key in required_keys:
            assert key in args, f"Missing required parameter: {key}"
            assert len(args[key]) > 0, f"Parameter {key} cannot be empty"

    def _load_annotations(self):
        """加载标注数据"""
        print("Loading annotations...")
        for vid in os.listdir(self.images_dir):
            vid_img_dir = os.path.join(self.images_dir, vid)
            if not os.path.isdir(vid_img_dir):
                continue

            frame_names = os.listdir(vid_img_dir)
            frame_names.sort()

            for frame_name in frame_names:
                if not frame_name.endswith('.jpg'):
                    continue

                gt_name = frame_name.replace(".jpg", ".txt")
                gt_path = os.path.join(self.gts_dir, vid, gt_name)

                try:
                    t = int(gt_name[:-4].split("-")[-1])
                except (ValueError, IndexError):
                    print(f"Warning: Cannot parse frame index from {gt_name}")
                    continue

                if os.path.exists(gt_path):
                    try:
                        with open(gt_path, 'r') as f:
                            for line in f:
                                line = line.strip()
                                if not line:
                                    continue
                                parts = line.split(" ")
                                if len(parts) < 6:
                                    print(f"Warning: Invalid annotation format in {gt_path}: {line}")
                                    continue
                                try:
                                    c, i = map(int, parts[:2])
                                    x, y, w, h = map(float, parts[2:6])
                                    self.gts[vid][t].append([c, i, x, y, w, h])
                                except ValueError:
                                    print(f"Warning: Cannot parse annotation in {gt_path}: {line}")
                                    continue
                    except Exception as e:
                        print(f"Error reading {gt_path}: {e}")
        print(f"Loaded annotations for {len(self.gts)} videos")

    def __getitem__(self, item):
        vid, begin_frame = self.sample_begin_frames[item]
        frame_idxs = self.sample_frames_idx(vid=vid, begin_frame=begin_frame)
        imgs, infos = self.get_multi_frames(vid=vid, idxs=frame_idxs)
        if self.transform is not None:
            imgs, infos = self.transform(imgs, infos)
        return {
            "imgs": imgs,
            "infos": infos
        }

    def __len__(self):
        assert self.sample_begin_frames is not None, "Please use set_epoch to init NuScenes Dataset."
        return len(self.sample_begin_frames)

    def sample_frames_idx(self, vid: str, begin_frame: int) -> list[int]:
        """采样帧索引，修正了类型注解"""
        if self.sample_mode == "random_interval":
            assert self.sample_length > 1, "Sample length must be greater than 1."
            remain_frames = self.sample_vid_tmax[vid] - begin_frame
            if remain_frames < self.sample_length - 1:
                # 如果剩余帧数不够，则连续采样
                frame_idxs = [begin_frame + _ for _ in range(self.sample_length)]
            else:
                max_interval = floor(remain_frames / (self.sample_length - 1))
                interval = min(randint(1, self.sample_interval), max_interval)
                frame_idxs = [begin_frame + interval * i for i in range(self.sample_length)]

                # 检查是否缺少标注文件
                is_lack = False
                for frame_idx in frame_idxs:
                    if frame_idx not in self.gts[vid]:
                        is_lack = True
                        break

                if is_lack:
                    frame_idxs = [begin_frame + _ for _ in range(self.sample_length)]

            return frame_idxs
        else:
            raise ValueError(f"Sample mode {self.sample_mode} is not supported.")

    def set_epoch(self, epoch):
        """设置当前epoch，更新采样策略"""
        self.sample_begin_frames = list()
        self.sample_vid_tmax = dict()
        self.sample_stage = 0

        # 根据epoch确定采样阶段
        for step in self.sample_steps:
            if epoch >= step:
                self.sample_stage += 1

        assert self.sample_stage < len(self.sample_steps) + 1, "Sample stage exceeds maximum stages"

        # 设置当前阶段的采样参数
        self.sample_length = self.sample_lengths[min(len(self.sample_lengths) - 1, self.sample_stage)]
        self.sample_mode = self.sample_modes[min(len(self.sample_modes) - 1, self.sample_stage)]
        self.sample_interval = self.sample_intervals[min(len(self.sample_intervals) - 1, self.sample_stage)]

        # 生成采样起始帧
        for vid in self.gts.keys():
            if not self.gts[vid]:  # 跳过空视频
                continue

            t_min = min(self.gts[vid].keys())
            t_max = max(self.gts[vid].keys())
            self.sample_vid_tmax[vid] = t_max

            for t in range(t_min, t_max - (self.sample_length - 1) + 1):
                # 检查连续帧是否都有标注
                filter_out = False
                for offset in range(self.sample_length):
                    if t + offset not in self.gts[vid]:
                        filter_out = True
                        break

                if not filter_out:
                    self.sample_begin_frames.append((vid, t))

        print(f"Epoch {epoch}: Sample stage {self.sample_stage}, "
              f"length={self.sample_length}, mode={self.sample_mode}, "
              f"interval={self.sample_interval}, total_samples={len(self.sample_begin_frames)}")

    def get_single_frame(self, vid, idx):
        """获取单帧数据"""
        img_path = os.path.join(self.images_dir, vid, f"{vid}-{idx:07d}.jpg")

        try:
            img = Image.open(img_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # 创建一个默认图像
            img = Image.new('RGB', (640, 480), color=(128, 128, 128))

        info = {
            "boxes": [],
            "ids": [],
            "labels": [],
            "areas": [],
            "frame_idx": torch.as_tensor(idx, dtype=torch.long)
        }

        # 添加标注信息
        for label, i, *xywh in self.gts[vid][idx]:
            # 验证边界框的有效性
            x, y, w, h = xywh
            if w > 0 and h > 0:  # 只添加有效的边界框
                info["boxes"].append([x, y, w, h])
                info["areas"].append(w * h)
                info["ids"].append(i)
                info["labels"].append(max(0, label - 1))  # 确保标签非负

        # 处理空标注的情况（添加假的标注以避免训练错误）
        if len(info["ids"]) == 0:
            info["boxes"].append([0.5, 0.5, 0.01, 0.01])  # 很小的假边界框
            info["areas"].append(0.0001)
            info["ids"].append(0)
            info["labels"].append(0)

        # 转换为tensor
        info["boxes"] = torch.as_tensor(info["boxes"], dtype=torch.float32)
        info["areas"] = torch.as_tensor(info["areas"], dtype=torch.float32)
        info["ids"] = torch.as_tensor(info["ids"], dtype=torch.long)
        info["labels"] = torch.as_tensor(info["labels"], dtype=torch.long)

        # 坐标转换：xywh to cxcywh (中心点格式)
        if len(info["boxes"]) > 0:
            boxes = info["boxes"].clone()
            # cx = x + w/2, cy = y + h/2
            boxes[:, :2] += boxes[:, 2:] / 2
            info["boxes"] = boxes

        return img, info

    def get_multi_frames(self, vid, idxs):
        """获取多帧数据"""
        frames_data = [self.get_single_frame(vid=vid, idx=i) for i in idxs]
        imgs, infos = zip(*frames_data)
        return list(imgs), list(infos)


def transforms_for_train():
    """训练时的数据变换"""
    # scales = [608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992]  # from MOTR
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]  # from COCO
    # NOTE: For NuScenes, we use 1333 instead of 1536, as the max size
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
            T.MultiNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    ])


# def transforms_for_val():
#     """验证时的数据变换"""
#     return T.MultiCompose([
#         T.MultiRandomResize([800], max_size=1333),
#         T.MultiCompose([
#             T.MultiToTensor(),
#             T.MultiNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])
#     ])


def build(args, split):
    """构建数据集"""
    if split == "train":
        return NuScenes(args, transforms_for_train(), split=split)
    # elif split == "val":
    #     # 如果需要验证集，可以创建对应的类
    #     return NuScenes(args, transforms_for_val(), split="train")  # 暂时使用train数据
    else:
        raise ValueError(f"Split {split} is not supported. Only 'train' and 'val' are supported.")
