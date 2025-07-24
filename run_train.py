# run_train.py
from train.trainer import Trainer
from train.config import parse_args
import os
import logging
import sys
from accelerate.utils import ProjectConfiguration
import datetime 

def _init_logging(rank, logging_dir):
    """
    初始化日志配置。
    该函数会根据进程的 rank（等级）来设置日志的输出方式。
    所有进程的日志都会输出到控制台，但只有 rank 0 的进程会将日志写入文件。
    这样做是为了避免在分布式训练中，多个进程同时写入同一个日志文件导致混乱。

    Args:
        rank (int): 当前进程在分布式环境中的全局排名。
        logging_dir (str): 日志文件的保存路径。
    """
    # 构建一个 handlers (处理器) 列表，用于定义日志的输出目标
    handlers = []

    # 添加一个 StreamHandler，将日志信息输出到标准输出流 (通常是控制台)
    handlers.append(logging.StreamHandler(stream=sys.stdout))

    # 只让 rank == 0 的主进程将日志写入文件，避免多进程重复写同一文件
    if rank == 0 and logging_dir:
        handlers.append(logging.FileHandler(logging_dir, mode='a', encoding='utf-8'))

    # 设置基础日志配置
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.ERROR,
        # 定义日志的输出格式：[时间] [日志级别] 日志消息
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        handlers=handlers
    )

    
def main():
    """
    程序的主入口函数。
    负责解析参数、初始化环境、创建 Trainer 对象并启动训练过程。
    """
    # 从yaml文件和命令行解析参数，来自于train.config文件
    args = parse_args()
    # 从环境变量中获取分布式训练相关的信息
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    os.makedirs(args.output_dir, exist_ok=True)

    #  创建一个唯一的文件名，例如 "train_2025-07-23_21-00-00.log"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"train_{timestamp}.log"
    log_file_path = os.path.join(args.output_dir, log_filename)
    _init_logging(rank,log_file_path)
    
     
    '''
    #创建一个 ProjectConfiguration 对象，用于告诉 Accelerator：你的训练工程的“输出路径”和“日志路径”分别在哪
    #project_dir : 整个训练工程的主目录，比如模型保存、检查点等输出会放这里
    #logging_dir: 所有日志相关内容的目录，例如 tensorboard 日志、wandb 日志等
    '''
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    args.accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    # 实例化 Trainer 类，并将所有配置参数 (args) 传入
    trainer = Trainer(args)
    # 调用 trainer 对象的 train 方法，开始模型的训练流程
    trainer.train()

if __name__ == "__main__":
    main()

