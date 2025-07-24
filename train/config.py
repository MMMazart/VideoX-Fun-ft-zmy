# config.py
import argparse
from omegaconf import OmegaConf, DictConfig
import sys

import argparse
import os
from omegaconf import OmegaConf, DictConfig
import sys

def parse_args() -> DictConfig:
    parser = argparse.ArgumentParser(description="VideoX-Fun-ft Training Configuration")

    parser.add_argument("--config", type=str, required=True, help="YAML 配置文件路径")
    parser.add_argument(
        "--override",
        nargs='*',
        default=[],
        help="命令行覆盖配置，例如：--override train.batch_size=8 model.use_lora=true"
    )

    args = parser.parse_args()

    # 检查配置文件是否存在
    if not os.path.exists(args.config):
        print(f"[错误] 配置文件不存在: {args.config}")
        sys.exit(1)

    # 加载 base 配置
    try:
        base_cfg = OmegaConf.load(args.config)
    except Exception as e:
        print(f"[读取失败] 配置文件 {args.config} 格式错误。\n错误信息: {e}")
        sys.exit(1)

    # 处理 override 参数
    if args.override:
        try:
            override_cfg = OmegaConf.from_dotlist(args.override)
        except Exception as e:
            print(f"[覆盖失败] 参数解析错误，请检查 --override 格式。\n示例: --override train.batch_size=8\n错误信息: {e}")
            sys.exit(1)
        
        # 验证所有 override 字段都存在于 base_cfg 中
        for key in override_cfg.keys():
            if OmegaConf.select(base_cfg, key) is None:
                print(f"[错误] 覆盖字段 `{key}` 在原始配置中不存在，请检查拼写。")
                sys.exit(1)
        
        # 合并配置
        final_cfg = OmegaConf.merge(base_cfg, override_cfg)
    else:
        final_cfg = base_cfg

    return final_cfg

# args = parse_args()
# print("dsadas")
# import pdb
# pdb.set_trace()