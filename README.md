# VideoX-Fun-ft-zmy
VideoX-Fun-ft-zmy/
│
├── README.md
├── run_train.py                     # 主训练脚本
│
├── configs/                         # 配置文件
│   ├── model_base.yaml              # 弃用
│   └── train_base.yaml              # 训练配置 + 模型配置
│
├── data/                            # 数据处理模块
│   ├── __init__.py
│   ├── collate.py                   # 数据聚合函数（例如batch构造）
│   ├── constants.py                 # 数据相关常量定义
│   ├── dataset.py                   # # 数据集读取与处理逻辑
│   ├── samplers.py                  # 数据采样器
│   ├── transforms.py                # 数据增强与变换
│   └── utils_data.py                # 数据相关的辅助函数
│
├── models/                          # 模型定义
│   ├── __init__.py
│   ├── cache_utils.py               # 缓存相关工具
│   ├── wan_image_encoder.py         # 图像编码器
│   ├── wan_text_encoder.py          # 文本编码器
│   ├── wan_transformer3d.py         # 3D Transformer 模块
│   ├── wan_vae.py                   # VAE 模块
│   └── wan_xlm_roberta.py           # XLM-RoBERTa 文本模型
│
├── train/                           # 训练流程封装
│   ├── __init__.py
│   ├── builder.py                   # 训练相关模块/组件构建器
│   ├── config.py                    # 配置解析，读取train_base.yaml和命令行参数
│   ├── engine.py                    # 训练引擎，封装单次训练
│   ├── hooks.py                     # 训练流程中的钩子，当前仅有acclerate的save_state
│   ├── losses.py                    # 损失函数
│   ├── optimizer.py                 # 优化器
│   ├── trainer.py                   # 训练主类
│   └── utils_train.py               # 训练辅助工具
│
└── utils/                           
    ├── discrete_sampler.py          # 离散采样器
    ├── lora_utils.py                # LoRA相关工具
    └── utils.py                     # 常规工具函数
