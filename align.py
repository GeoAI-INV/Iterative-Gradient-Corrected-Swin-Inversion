# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 16:20:06 2024
@author: Qi Pang
"""

import argparse
from datetime import datetime
import torch
import matplotlib.pyplot as plt
from scipy.io import loadmat
import time
from scipy.io import savemat
from numpy import random
import numpy as np
import os
import joblib
import torch
import segyio
from utils.functions import get_low, get_patches, back_image
from utils.MakeRecord import forwardblock
from scipy.ndimage import gaussian_filter,gaussian_filter1d
from scipy.linalg import lstsq
import pylops
from matplotlib.ticker import LogFormatterSciNotation
from scipy.linalg import toeplitz
import torch.nn.functional as F

random_seed = 42
np.random.seed(random_seed)

plt.rcParams['figure.figsize'] = [8, 6]
plt.rcParams['figure.dpi'] = 100
plt.rcParams['figure.facecolor'] = 'w'
plt.rcParams['font.size'] = 18
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['lines.linewidth'] = 2


def align_process(matrix):
    "拉平操作 对0元素操作"
    rows, cols = matrix.shape
    result = np.zeros_like(matrix)
    max_length = 0
    for col in range(cols):
        nonzero_indices = np.where(matrix[:, col] != 0)[0]
        if nonzero_indices.size > 0:
            max_length = max(max_length, len(nonzero_indices))
    for col in range(cols):
        current_col = matrix[:, col]
        nonzero_indices = np.where(current_col != 0)[0]
        if nonzero_indices.size > 0:
            first_nonzero_index = nonzero_indices[0]
            nonzero_length = len(nonzero_indices)
            result[:rows - first_nonzero_index, col] = current_col[first_nonzero_index:]
            if nonzero_length < max_length:
                last_value = result[rows - first_nonzero_index - 1, col]
                result[rows - first_nonzero_index:rows - first_nonzero_index + (max_length - nonzero_length), col] = last_value
    return result


def restore_original_matrix(modified_matrix, original_matrix):
    "反拉平"
    rows, cols = original_matrix.shape
    result = np.zeros_like(original_matrix)

    for col in range(cols):
        current_col = modified_matrix[:, col]
        nonzero_indices = np.where(original_matrix[:, col] != 0)[0]
        if nonzero_indices.size > 0:
            first_nonzero_index = nonzero_indices[0]
            nonzero_length = len(nonzero_indices)
            result[first_nonzero_index:first_nonzero_index + nonzero_length, col] = current_col[:nonzero_length]

    return result


def fill_zeros_with_nearest_nonzero(matrix):
    "填充"
    rows, cols = matrix.shape
    result = matrix.copy()
    for col in range(cols):
        for row in range(rows):
            if result[row, col] == 0:
                for i in range(row - 1, -1, -1):
                    if result[i, col] != 0:
                        result[row, col] = result[i, col]
                        break
                if result[row, col] == 0:
                    for i in range(row + 1, rows):
                        if result[i, col] != 0:
                            result[row, col] = result[i, col]
                            break
    return result


field_syn = loadmat('field_data/field_syn.mat')['field_syn']  # 未拉平的原始地震数据 这里面没有值的地方是0
field_imp = loadmat('field_data/field_imp.mat')['field_imp']  # 未拉平的原始阻抗数据 这里面没有值的地方是
field_imp[field_imp == 0] = 1  # 保证计算对数阻抗后再0的位置仍未0 log(1)=0

imp = np.log(field_imp)  #对数阻抗

well1 = 38
well2 = 54
well3 = 68
well4 = 90

imp1 = align_process(imp)  # 这个函数是拉平操作 将每一道的数据拉到最上面(顶部对齐) 但是底部未对齐 
imp1 = imp1[:94, :] # 手动选的到哪里截止 因为第94行之后全是0 根据数据选择
imp1 = fill_zeros_with_nearest_nonzero(imp1) # 因为每一道的数据长度可能不一样 在最低端可能还有0 将这些0值通过当前到的最近元素填充
syn1 = align_process(field_syn)  #地震数据同样的操作
syn1 = syn1[:94,:]
syn1 = fill_zeros_with_nearest_nonzero(syn1)
# imp1 和 syn1是对齐后的数据

imp_back = restore_original_matrix(imp1, imp) # 反拉平操作 将拉平的imp1通过和未拉平的原始imp对比 得到反拉平的imp_back  这里imp只需要未拉平的位置 用syn也可以因为他们位置信息一样
syn_back = restore_original_matrix(syn1, imp) 

m = syn_back-field_syn # 检验
print(np.max(abs(m)))




