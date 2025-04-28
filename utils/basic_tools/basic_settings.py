"""
Created on Mon Mar 24 09:58:22 2025
@author: Qi Pang
Theme: some basic settings such as random_set
"""
import matplotlib.pyplot as plt
import os
import random
import warnings
import numpy as np
import torch

def set_deterministic_seed(seed=42):
    """
    设置随机种子以确保实验可复现性
    """
    if not isinstance(seed, int):
        raise TypeError(f"种子必须是整数类型，但传入的是 {type(seed)} 类型")

    os.environ['PYTHONHASHSEED'] = str(seed)  # 设置Python哈希种子
    random.seed(seed)  # 设置Python内置随机数
    np.random.seed(seed)  # 设置NumPy随机种子
    torch.manual_seed(seed)  # 设置PyTorch随机种子
    # CUDA相关设置
    if torch.cuda.is_available():
        # 设置所有CUDA设备的种子
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多GPU情况

        torch.backends.cudnn.enabled = True  # GPU加速, default is True
        # 配置cuDNN以实现确定性
        torch.backends.cudnn.benchmark = False  # True, PyTorch会根据输入大小自动寻找最优的卷积算法
        torch.backends.cudnn.deterministic = True

        if not torch.backends.cudnn.deterministic:
            warnings.warn("当前cuDNN版本不支持确定性模式！结果可能不可复现")
    else:
        warnings.warn("CUDA不可用，跳过GPU相关种子设置")
    print(f"cuDNN_deterministic: {torch.backends.cudnn.deterministic}")


