"""
Created on Mon Jun 17 17:02:43 2024
@author: Pang Qi
"""
import argparse
from datetime import datetime
import torch
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
import time
from numpy import random
import numpy as np
import os
import joblib
import torch
import segyio
from utils.functions import get_low, get_patches, back_image
from utils.MakeRecord import forwardblock
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.linalg import lstsq
import pylops
from matplotlib.ticker import LogFormatterSciNotation
from scipy.linalg import toeplitz
import torch.nn.functional as F

vpk = loadmat('Data/vp.mat')['vp'][500:500 + 551, 0:13601:10]
denk = loadmat('Data/den.mat')['den'][500:500 + 551, 0:13601:10]
logimp = np.log(vpk * denk)

implow = get_low(logimp, 100, 50)
plt.figure()
plt.imshow(implow, 'jet', aspect='auto')
plt.tight_layout()
plt.show()

random_seed = 42
np.random.seed(random_seed)

plt.rcParams['figure.figsize'] = [8, 6]
plt.rcParams['figure.dpi'] = 100
plt.rcParams['figure.facecolor'] = 'w'
plt.rcParams['font.size'] = 18
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['lines.linewidth'] = 2


def equalize_nonzero_lengths(matrix):
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
    rows, cols = matrix.shape
    result = matrix.copy()
    for col in range(cols):
        for row in range(rows):
            if result[row, col] == 0:
                # Find the nearest non-zero element above
                for i in range(row - 1, -1, -1):
                    if result[i, col] != 0:
                        result[row, col] = result[i, col]
                        break
                # If no non-zero element above, find the nearest non-zero element below
                if result[row, col] == 0:
                    for i in range(row + 1, rows):
                        if result[i, col] != 0:
                            result[row, col] = result[i, col]
                            break
    return result

def estimate_wavelet_lstsq(seis, r, wave_len):
    """seis: seismic data, r: reflectivity, wave_len: wavelet length"""
    op = np.zeros((r.size + wave_len - 1, wave_len))
    for i in range(wave_len):
        op[i:r.size + i, i] = r
    op_1 = op[wave_len // 2 + 1:wave_len // 2 + r.size + 1, :]
    wave_inv = lstsq(op_1, seis)[0]
    return wave_inv


def opConvolve(data, kernel_size):
    op = np.zeros((data.size + kernel_size - 1, kernel_size))
    for i in range(kernel_size):
        op[i:data.size + i, i] = data
    op_1 = op[kernel_size // 2 + 1:kernel_size // 2 + data.size + 1, :]
    return op_1


syn = loadmat('field_data/field_syn.mat')['field_syn']
field_imp = loadmat('field_data/field_imp.mat')['field_imp']
field_imp[field_imp == 0] = 1
# filled_z = fill_zeros_with_nearest_nonzero(field_imp)
imp = np.log(field_imp)

well1 = 38
well2 = 54
well3 = 68
well4 = 90

imp1 = equalize_nonzero_lengths(imp)
imp1 = imp1[:94, :]
imp1 = fill_zeros_with_nearest_nonzero(imp1)
syn1 = equalize_nonzero_lengths(syn)
syn1 = syn1[:94, :]
syn1 = fill_zeros_with_nearest_nonzero(syn1)
m = get_patches(imp1, [64, 64], [1, 1])
# savemat(f'{"field_data"}/{"field_imp_align"}.mat', {"field_imp_align": imp1})
# savemat(f'{"field_data"}/{"field_syn_align"}.mat', {"field_syn_align": syn1})
k = -3


def DIFFZ(z):
    DZ = np.zeros_like(z)
    # DZ[0:-1,:] = 0.5 * (z[1:,:] - z[:-1,:])
    DZ[1:-1, :] = 0.5 * (z[2:, :] - z[:-2, :])
    return DZ


z = DIFFZ(imp1)

# C = opConvolve(z[:,well1], 60)
# wave_inv = lstsq(C, syn[:,well1])[0]
wave_inv = estimate_wavelet_lstsq(syn1[:, well1], z[:, well1], 61)
# wave_inv = estimate_wavelet_lstsq(z[:,well1],)
# C = opConvolve(z[71-k:158+1+k,well3], 60)
# wave_inv = lstsq(C, syn[71-k:158+1+k,well3])[0]
wave_inv = gaussian_filter1d(wave_inv, 1)


# vpk = loadmat('.//Data//vp.mat')['vp'][500:500+551, 0:13601:10]
# denk = loadmat('.//Data//den.mat')['den'][500:500+551, 0:13601:10]
# logimp = np.log(vpk * denk)
# z = get_patches(logimp , [128,128], [30,30])
# syn_model = forwardblock(logimp, f0=30, dt=0.001, wave_len=41, snr_db=10).cleansyn()
# real_ricker = forwardblock(logimp, f0=30, dt=0.001, wave_len=41, snr_db=10).wavelet()
# z_logimp = DIFFZ(logimp)
# C = opConvolve(z_logimp[200:300,68].T, 81)
# wave_ricker = lstsq(C, syn_model[200:300,68].T)[0]
# X=np.zeros((len(z_logimp[:,68]),1))
# d=np.append(wave_ricker.reshape(len(wave_ricker), 1),X,axis=0)
# W_temp = toeplitz(d, np.zeros(len(z_logimp[:,68])))
# WW = W_temp[(len(real_ricker) - 1) // 2 : - (len(real_ricker) - 1) // 2 -1, :]
# # syn_model_1 = np.dot(C, wave_ricker)
# # syn_model1 = pylops.avo.poststack.PoststackLinearModelling(wave_ricker, nt0=logimp.shape[0], spatdims=logimp.shape[1]) * logimp

def DataProess(data):
    data = data / np.max(abs(data))
    patch_data = np.expand_dims(get_patches(data, [64, 64], [1, 1]), 1)
    patch_data = torch.tensor(patch_data).type(torch.float32)
    return patch_data


def conv_1D(wav, r, s):
    WEIGHT = torch.tensor(wav.reshape(1, 1, wav.shape[0])).type(torch.float32)
    For_syn = torch.zeros_like(r)
    for i in range(r.shape[3]):
        INPUT = r[..., i]
        For_syn[..., i] = F.conv1d(INPUT, WEIGHT, stride=1, padding=(len(wav) - 1) // 2)
    return For_syn * s


def Pylop_diff(z):
    DZ = torch.zeros([z.shape[0], z.shape[1], z.shape[2], z.shape[3]]).type(torch.float32)
    DZ[..., 1:-1, :] = 0.5 * (z[..., 2:, :] - z[..., :-2, :])
    return DZ


s_min_max = [np.min(abs(syn1)), np.max(abs(syn1))]
m_min_max = [np.min(abs(imp1)), np.max(abs(imp1))]

patch_syn = DataProess(syn1)
syn_nor = syn1 / np.max(abs(syn1))
patch_logimp = DataProess(imp1)

pytorch_syn = conv_1D(wave_inv, Pylop_diff(patch_logimp), m_min_max[1] / s_min_max[1])
torch_syn = pytorch_syn.numpy()
torch_syn = back_image(torch_syn.squeeze(), [94, 110], [1, 1])

m = loadmat('field_data//field_syn_pwd.mat')['field_syn_pwd']

syn_py1 = pylops.avo.poststack.PoststackLinearModelling(wave_inv, nt0=imp1.shape[0], spatdims=imp1.shape[1]) * imp1

pre_imp = loadmat('field_data//field_predict.mat')['field_predict']

imp_re = restore_original_matrix(pre_imp, imp)
imp_k = np.exp(imp_re)
imp_k[imp_k == 1] = 0
# savemat(f'{"field_data"}/{"field_imp_predict"}.mat', {"field_imp_predict": imp_k})
scales = 0.05 * imp.shape[0] / imp.shape[1]

# savemat(f'{"field_data"}/{"field_imp_predict_1"}.mat', {"field_imp_predict_1": pre_imp})
# vmin = np.log(4100)
# vmax = np.log(7900)

# plt.figure(figsize=(8, 4))
# plt.imshow(imp1, 'jet', aspect='auto', vmin=8.1, vmax=9.4)
# plt.tight_layout()
# cbar = plt.colorbar(fraction=scales, pad=0.02)
# plt.show()


cmap = plt.get_cmap('jet')
# cmap.set_bad(color='white')

# Mask the zero values in the data
imp_re_masked = np.ma.masked_where(imp_re == 0, imp_re)
data = np.exp(imp_re_masked)
vmin = 6150
vmax = 7550
col_index = 58  
column_values = data[:, col_index]
fig, ax = plt.subplots()
plt.imshow(data, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
plt.tight_layout()
cbar = plt.colorbar(fraction=scales, pad=0.02)
plt.show()
# true_imp= field_imp
# true_imp[field_imp == 1] = 6600
# cdp = [38, 68, 90, 53]
# # cdp = [38, 68, 90, 53, 51, 52, 54, 55]
# for i in cdp:
#     fig, ax = plt.subplots(figsize=(4, 13), layout='constrained')
#     ax.plot(true_imp[:, i], np.arange(true_imp.shape[0]),color='k')
#     # ax.plot(imp_re[:, i], label='True',color='k')
#     # ax.plot(implow[:, i], label='LowF')
#     # ax.plot(imp_0[:, i], label=f'ep50 (PCC: {pcc0:.4f})')
#     # ax.plot(imp_1[:, i], label=f'ep100 (PCC: {pcc1:.4f})')
#     # ax.plot(imp_2[:, i], label=f'ep150 (PCC: {pcc2:.4f})')
#     # ax.plot(imp_3[:, i], label=f'ep200 (PCC: {pcc3:.4f})')
#     ax.set_xlabel('Time (ms)')
#     ax.set_ylabel(r'Impedance ($\mathrm{g/cm^3 \cdot km/s}$)')
#     # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.33), ncol=4, frameon=False)  # Legend above the plot
#     ax.set_title(f'CDP {i + 1} AI', fontsize=14)
#     ax.invert_yaxis()  # Invert the y-axis
#     # ax.set_xlim(0, true_imp.shape[0] - 1)
#     plt.show()
implow = gaussian_filter(imp1, sigma=10)
# implow = get_low(imp1, 20, 30)
implow = restore_original_matrix(implow, imp)
imp_re_masked = np.ma.masked_where(implow == 0, implow)
plt.figure(figsize=(8, 4))
plt.imshow(np.exp(imp_re_masked), cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
plt.tight_layout()
cbar = plt.colorbar(fraction=scales, pad=0.02)
plt.show()

# z=np.ma.masked_where(syn == 0, syn)
# plt.figure(figsize=(8, 4))
# plt.imshow(np.ma.masked_where(syn == 0, syn), 'seismic', aspect='auto')
# plt.tight_layout()
# cbar = plt.colorbar(fraction=scales, pad=0.02)
# plt.show()

# plt.figure(figsize=(8, 4))
# plt.imshow(syn_nor, 'seismic', aspect='auto')
# plt.tight_layout()
# cbar = plt.colorbar(fraction=scales, pad=0.02)
# plt.show()

# plt.figure(figsize=(8, 4))
# plt.imshow(torch_syn, 'seismic', aspect='auto')
# plt.tight_layout()
# cbar = plt.colorbar(fraction=scales, pad=0.02)
# plt.show()

# plt.figure(figsize=(8, 4))
# plt.imshow(syn_py1, 'seismic', aspect='auto',vmin=np.min(syn), vmax=np.max(syn))
# plt.tight_layout()
# cbar = plt.colorbar(fraction=scales, pad=0.02)
# plt.show()

# cdp = [5,10,20,30,38,54,68,90]
# for i in cdp:
#     fig, ax = plt.subplots(figsize=(8, 4))
#     ax.plot(syn_nor[:, i], label='real seismic')
#     ax.plot(torch_syn[:, i], label='forward based estimated wav')
#     ax.set_xlabel('Time (ms)')
#     ax.legend()
#     ax.set_title(f'CDP {i}')
#     plt.tight_layout()  
#     plt.show()


# fig, ax = plt.subplots(figsize=(8, 4))
# ax.plot(wave_inv, label='wavelet')
# # ax.plot(m, label='real')
# # ax.plot(wave_inv, label='estimate')
# ax.legend()
# plt.tight_layout() 
# plt.show()

# fig, ax = plt.subplots(figsize=(8, 4))
# ax.plot(imp1[:,0], label='real')
# ax.plot(implow[:,0], label='estimate')
# ax.legend()
# plt.tight_layout()  
# plt.show()

# bar_title = r'Impedance ($\mathrm{g/cm^3 \cdot km/s}$)'
# plt.figure(figsize=(4, 4), layout='constrained')
# plt.imshow(imp1, 'jet')
# plt.xlabel('CDP')
# plt.ylabel('Time (ms)')
# cbar = plt.colorbar(fraction= 0.05 * imp1.shape[0] / imp1.shape[1], pad=0.02)
# cbar.set_label(bar_title)

