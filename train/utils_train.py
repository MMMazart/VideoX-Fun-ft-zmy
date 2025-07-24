import logging
def get_logger(name = None):
    logger = logging.getLogger(name or __name__)
    logger.setLevel(logging.INFO)
    return logger


def filter_kwargs(cls, kwargs):
    """
    一个工具函数，用于过滤字典，只保留那些类 `cls` 的构造函数 `__init__` 中存在的参数。
    这在从一个大的配置文件中为特定类提取参数时非常有用。
    """
    import inspect
    sig = inspect.signature(cls.__init__)
    # 获取构造函数的所有有效参数名
    valid_params = set(sig.parameters.keys()) - {'self', 'cls'}
    # 从输入的 kwargs 中筛选出有效的参数
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    return filtered_kwargs

