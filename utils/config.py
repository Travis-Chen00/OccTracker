import os
import yaml
from yacs.config import CfgNode as CN


def load_config(config_path):
    """
    Load configuration from a YAML file

    Args:
        config_path (str): Path to the configuration YAML file

    Returns:
        CN: Configuration object
    """
    # Create a default config
    _C = CN()

    # Load from YAML
    with open(config_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)

    # Update config
    _C.update(cfg_dict)

    return _C


def update_config(config, args=None, **kwargs):
    """
    Update configuration with additional parameters

    Args:
        config (CN): Original configuration
        args (Namespace or dict, optional): Arguments to update
        **kwargs: Additional key-value pairs to update

    Returns:
        CN: Updated configuration
    """
    config.defrost()

    # 处理args（支持Namespace和dict）
    if args is not None:
        # 将Namespace转换为字典
        if hasattr(args, '__dict__'):
            args_dict = vars(args)
        else:
            args_dict = args

        for key, value in args_dict.items():
            # 忽略None值
            if value is not None:
                # 检查顶层配置
                if hasattr(config, key.upper()):
                    setattr(config, key.upper(), value)

                # 检查嵌套配置
                nested_sections = ['DATA', 'MODEL', 'TRAIN', 'DATASET']
                for section in nested_sections:
                    if hasattr(config, section):
                        section_config = getattr(config, section)
                        if hasattr(section_config, key.upper()):
                            setattr(section_config, key.upper(), value)

    # 处理直接传入的关键字参数
    for key, value in kwargs.items():
        if value is not None:
            if hasattr(config, key.upper()):
                setattr(config, key.upper(), value)

            # 检查嵌套配置
            nested_sections = ['DATA', 'MODEL', 'TRAIN', 'DATASET']
            for section in nested_sections:
                if hasattr(config, section):
                    section_config = getattr(config, section)
                    if hasattr(section_config, key.upper()):
                        setattr(section_config, key.upper(), value)

    config.freeze()
    return config



# def save_config(config, save_path):
#     """
#     Save configuration to a YAML file
#
#     Args:
#         config (CN): Configuration to save
#         save_path (str): Path to save the configuration
#     """
#     config.defrost()
#     # 转换为普通字典
#     config_dict = {}
#     for k, v in config.items():
#         if isinstance(v, CN):
#             config_dict[k] = {inner_k: inner_v for inner_k, inner_v in v.items()}
#         else:
#             config_dict[k] = v
#
#     with open(save_path, 'w') as f:
#         yaml.dump(config_dict, f, default_flow_style=False)
#     config.freeze()


# 使用示例
# if __name__ == "__main__":
#     # 加载配置
#     config_path = "/home/boyu/Desktop/OccTracker/config/tracker.yaml"
#     cfg = load_config(config_path)
#
#     # 方法1：传入字典
#     args_dict = {
#         'batch_size': 16,
#         'lr': 1e-3,
#         'num_workers': 4,
#         'version': 'v1.0-trainval'
#     }
#     updated_cfg = update_config(cfg, args=args_dict)
#
#     # 打印更新后的配置
#     # print("Updated Batch Size:", updated_cfg.BATCH_SIZE)
#     # print("Updated Learning Rate:", updated_cfg.LR)
#     # print("Updated Version:", updated_cfg.VERSION)
#     print(updated_cfg)
#
#     # 保存更新后的配置
#     # save_config(updated_cfg, "/config/updated_model.yaml")
