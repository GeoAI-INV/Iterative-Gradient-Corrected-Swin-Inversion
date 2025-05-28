import math
import os
from os.path import isdir

import joblib
import numpy as np
import pywt
import torch
import torch.nn.functional as F
from pylops.utils.wavelets import ricker
from pylops.utils.signalprocessing import dip_estimate
from scipy.io import loadmat
from scipy.linalg import lstsq
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.stats import pearsonr
from skimage.metrics import peak_signal_noise_ratio as psnr
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from collections import OrderedDict
from obspy.io.segy.segy import _read_segy

from NetModel.MyTransformer.TransNet import STMNet_1
from NetModel.Net2D import MyNet
from NetModel.test_paper import CLWTNet
from utils.MakeRecord import forwardblock
from utils.functions import get_low, get_patches, back_image, add_gaussian_noise, batch_snr, \
    denor_data, evaluation

dtype = torch.cuda.FloatTensor
device = torch.device("cuda:0")


# --------------------------------------------------------------------------------------------------------
def DIFFZ(z):
    DZ = np.zeros_like(z)
    DZ[1:-1, :] = 0.5 * (z[2:, :] - z[:-2, :])
    return DZ


# ----------------------------------------------------------------------------------------------------------------------
def single_snr(signal, noise_data):
    signal = np.array(signal.squeeze())
    noise_data = np.array(noise_data.squeeze())
    p_signal = np.mean(signal ** 2)
    p_noise = np.mean((noise_data - signal) ** 2)
    if p_noise == 0:
        return 999
    snr = 10 * math.log10(p_signal / p_noise)
    return snr


# ----------------------------------------------------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------------------------------------------------------------------
def get_cfs(data, scales):
    cfs = np.stack([np.real(pywt.cwt(data[:, i], scales, 'gaus2', 0.001)[0]) for i in
                    range(data.shape[-1])], axis=-1)
    f_cfs = pywt.cwt(data[:, 0], scales, 'gaus2', 0.001)[1]
    ricker_wavelets = []
    for i in f_cfs:
        ricker_wavelets.append(ricker(np.arange(51) * 0.001, i)[0])
    return cfs, ricker_wavelets


# ------------------------------------------------------------------------------------------------------------------------------------------
def estimate_wavelet_lstsq(seis, r, wave_len):
    op = np.zeros((r.size + wave_len - 1, wave_len))
    for i in range(wave_len):
        op[i:r.size + i, i] = r
    op_1 = op[wave_len // 2 + 1:wave_len // 2 + r.size + 1, :]
    wave_inv = lstsq(op_1, seis)[0]
    return wave_inv


# ------------------------------------------------------------------------------------------------------------------------------------------
def smooth_v1(data, sigma=2):
    data_de = np.zeros_like(data)
    for i in range(data.shape[1]):
        data_de[:, i] = gaussian_filter1d(data[:, i], sigma)
    return data_de


# ------------------------------------------------------------------------------------------------------------------------------------------
def conv_1D(wav, r, s):
    WEIGHT = torch.tensor(wav.reshape(1, 1, wav.shape[0])).to(device).type(torch.float32)
    For_syn = torch.zeros_like(r)
    for i in range(r.shape[3]):
        INPUT = r[..., i]
        For_syn[..., i] = F.conv1d(INPUT, WEIGHT, stride=1, padding=(len(wav) - 1) // 2)
    return For_syn * s


# ----------------------------------------------------------------------------------------------------------------------
def Pylop_diff(z):
    DZ = torch.zeros([z.shape[0], z.shape[1], z.shape[2], z.shape[3]]).to(device).type(torch.float32)
    DZ[..., 1:-1, :] = 0.5 * (z[..., 2:, :] - z[..., :-2, :])
    return DZ


def tv_loss(x, weight, weight1):
    # Compute differences
    dw = torch.abs(torch.gradient(x, dim=3)[0])
    dh = torch.abs(torch.gradient(x, dim=2)[0])
    # Combine and average
    return torch.mean(weight * dw) + torch.mean(weight1 * dh)


# ----------------------------------------------------------------------------------------------------------------------
def dip_loss(x, angle, args, weight, weight1):
    dip_cos = torch.cos(torch.deg2rad(angle))
    dip_sin = torch.sin(torch.deg2rad(angle))
    dx = torch.gradient(x, dim=3)[0]
    dz = torch.gradient(x, dim=2)[0]
    d1 = dip_cos * dx + dip_sin * dz  
    d2 = -dip_sin * dx + dip_cos * dz  
    return torch.mean(weight * torch.abs(d1)) + torch.mean(weight1 * torch.abs(d2))


# ----------------------------------------------------------------------------------------------------------------------
def calculate_angle(record):
    rx = np.diff(record, axis=1)
    rx = rx[:-1, :]
    rz = np.diff(record, axis=0)
    rz = rz[:, :-1] + 1e-6
    angle = np.zeros_like(record)
    angle[:-1, :-1] = np.rad2deg(np.arctan(rx / rz))
    return -angle


# ----------------------------------------------------------------------------------------------------------------------
def LoadData(args):
    if args.seam:
        segy = _read_segy("./Data/SEAM_I_2D_Model/SEAM_Vp_Elastic_N23900.sgy")
        vpk = [trace.data for trace in segy.traces]
        vpk = np.array(vpk).transpose()[::3, ::2][55:-40, :]
        segy = _read_segy("./Data/SEAM_I_2D_Model/SEAM_Vs_Elastic_N23900.sgy")
        denk = [trace.data for trace in segy.traces]
        denk = np.array(denk).transpose()[::3, ::2][55:-40, :]
        print('seam')
    else:
        vpk = loadmat('.//Data//vp.mat')['vp'][500:500 + args.Nt, 0:13601:10]
        denk = loadmat('.//Data//den.mat')['den'][500:500 + args.Nt, 0:13601:10]
        print('mar')
    logimp = np.log(vpk * denk)
    if args.stage == 1:
        implow = get_low(logimp, 400, 180)
    else:
        print('stage2')
        implow = loadmat('.//test_advan//complex_trans_mlow_100ep.mat')['complex_trans_mlow_100ep']
    synblock = forwardblock(logimp, f0=30, dt=0.001, wave_len=51)
    if args.use_noise:
        data = synblock.noisesyn(snr_db=args.snr)
        data = smooth_v1(data, 2)  # for estimating dip
    else:
        data = synblock.cleansyn()
    if args.reg == 'tv':
        weight = args.mu * np.ones_like(logimp)
        weight1 = args.mu1 * np.ones_like(logimp)
        print('tv weight')
    elif args.reg == 'dip':
        weight = args.hor * np.ones_like(logimp)
        weight1 = args.ver * np.ones_like(logimp)
        print('dip weight')
    else:
        weight = np.ones_like(logimp)
        weight1 = np.ones_like(logimp)
        print('no weight')

    if args.angle == 'direct':
        angle = calculate_angle(data)
        print('direct')
    else:
        if args.snr == 15:
            # change smooth with noise level
            angle = -np.degrees(dip_estimate(data, smooth=3)[0])
            print('pwd 15')
        elif args.snr == 10:
            angle = -np.degrees(dip_estimate(data, smooth=3)[0])
            print('pwd 10')
        elif args.snr == 5:
            angle = -np.degrees(dip_estimate(data, smooth=3)[0])
            print('pwd 5')
        elif args.snr == 3:
            angle = loadmat('.//Data//angle_pwd_3db.mat')['angle_pwd_3db'][:-1, :]
            print('pwd 3')
        elif args.snr == 1:
            angle = -np.degrees(dip_estimate(data, smooth=3)[0])
            print('pwd 1')
    scales = [6, 7.5, 15, 30]
    cfs, cfs_wavelets = get_cfs(data, scales)
    labels = np.zeros_like(logimp)
    labels[:, args.labels] = 1
    return logimp, implow, synblock, labels, cfs, cfs_wavelets, angle, weight, weight1


def LoadData_field(args):
    aligned_logimp = loadmat('.//field_data//field_imp_align.mat')['field_imp_align']
    aligned_syn = loadmat('.//field_data//field_syn_align.mat')['field_syn_align']
    if args.align:
        print('aligned')
        logimp = loadmat('.//field_data//field_imp_align.mat')['field_imp_align']
        syn = loadmat('.//field_data//field_syn_align.mat')['field_syn_align']
    else:
        print('unaligned')
        imp = loadmat('.//field_data//field_imp.mat')['field_imp']
        imp = fill_zeros_with_nearest_nonzero(imp)
        logimp = np.log(imp)
        syn = loadmat('.//field_data//field_syn.mat')['field_syn']
    r = DIFFZ(aligned_logimp)
    wavelet = estimate_wavelet_lstsq(aligned_syn[:, args.labels[0]], r[:, args.labels[0]], 61)

    if args.reg == 'tv':
        weight = args.mu * np.ones_like(logimp)
        weight1 = args.mu1 * np.ones_like(logimp)
        print('tv weight')
    elif args.reg == 'dip':
        weight = args.hor * np.ones_like(logimp)
        weight1 = args.ver * np.ones_like(logimp)
        print('dip weight')
    else:
        weight = np.zeros_like(logimp)
        weight1 = np.zeros_like(logimp)
        print('no weight')

    # wavelet = gaussian_filter1d(wavelet, 1)
    if args.stage == 1:
        implow = gaussian_filter(logimp, sigma=10) 
        # implow = restore_original_matrix(implow, syn)
        # implow = fill_zeros_with_nearest_nonzero(implow)
    else:
        print('stage2')
        implow = loadmat('.//test_advan//complex_trans_mlow_100ep.mat')['complex_trans_mlow_100ep']
    if args.use_noise:
        data = add_gaussian_noise(syn, args.snr_db)
    else:
        data = syn
    if args.angle == 'direct':
        angle = calculate_angle(data)
        print('direct')
    else:
        # angle = loadmat('.//field_data//field_syn_pwd.mat')['field_syn_pwd'][:-1, :]
        angle = -np.degrees(dip_estimate(data, smooth=3)[0])
        print('pwd')
    scales = [6, 7.5, 15, 30]
    cfs, cfs_wavelets = get_cfs(data, scales)  # 第一个数据是高频数据
    # labels trace
    labels = np.zeros_like(logimp)
    labels[:, args.labels] = 1
    print(logimp.shape, data.shape)
    return logimp, implow, data, labels, cfs, cfs_wavelets, angle, wavelet, weight, weight1


# ----------------------------------------------------------------------------------------------------------------------
def LoadData_v2():
    logimp = joblib.load('.//Data//imp_train.pkl')
    syn_train = joblib.load('.//Data//syn_train.pkl')
    syn_noise = joblib.load('.//Data//syn_noise_10db.pkl')
    Masks = joblib.load('.//Data//Masks.pkl')
    implog_train = joblib.load('.//Data//implog_train.pkl')

    imp_test = joblib.load('.//Data//impedance_model.pkl')  
    syn_test = joblib.load('.//Data//syn_freenoise.pkl')

    implow = []
    for x in implog_train:
        implow.append(get_low(x, 10, 50))

    imp_test = [imp_test[:, i, :] for i in range(imp_test.shape[1])] 
    syn_test = [syn_test[:, i, :] for i in range(syn_test.shape[1])]
    syn_test_noise = joblib.load('.//Data//syn_test_10db.pkl')
    low_test = []

    for x in imp_test:
        low_test.append(get_low(x, 10, 50))

    return logimp, implow, syn_train, syn_noise, implog_train, Masks, imp_test, syn_test, syn_test_noise, low_test


# ----------------------------------------------------------------------------------------------------------------------
def DataProess(data, args):
    # maxdata = np.tile(np.max(abs(data)), data.shape)
    # data = normalize_data(data)
    patch_data = np.expand_dims(get_patches(data, args.patchsize, args.stride), 1)
    # patch_maxdata = np.expand_dims(get_patches(maxdata, args.patchsize, args.stride), 1)
    patch_data = torch.tensor(patch_data).to(device).type(torch.float32)
    # patch_maxdata = torch.tensor(patch_maxdata).to(device).type(torch.float32)
    return patch_data

# ----------------------------------------------------------------------------------------------------------------------
def get_data(args, test=False):
    # Loading data
    if args.data_flag == 'model':
        print('model')
        logimp, implow, synblock, labels, cfs, cfs_wavelets, angle, weight, weight1 = LoadData(args)
        wav = synblock.wavelet()
        if args.use_noise:
            syn = synblock.noisesyn(snr_db=args.snr, mode=args.noise_type)
            print('noise')
        else:
            syn = synblock.cleansyn()
            print('clean')
    else:
        print('field')
        logimp, implow, syn, labels, cfs, cfs_wavelets, angle, wav, weight, weight1 = LoadData_field(args)

    assert logimp.shape[1] == syn.shape[
        1], 'Number of traces is not consistent. Got {} traces for seismic data and {} traces for acoustic acoustic impedance'.format(
        logimp.shape[0], syn.shape[0])

    s_min_max = [np.min(abs(syn)), np.max(abs(syn))]
    m_min_max = [np.min(abs(logimp)), np.max(abs(logimp))]
    mlow_min_max = [np.min(abs(implow)), np.max(abs(implow))]
    cfs_min_max = [np.min(abs(cfs[-1, :, :])), np.max(abs(cfs[-1, :, :]))]
    high_min_max = [np.min(abs(cfs[0, :, :])), np.max(abs(cfs[0, :, :]))]

    # 归一化方式
    syn = syn / np.max(abs(syn))
    logimp = logimp / np.max(abs(logimp))
    implow = implow / np.max(abs(implow))
    cfs /= np.max(np.abs(cfs), axis=(1, 2), keepdims=True)
    # print()

    # 另一种归一化方式
    # syn = 2 * (syn - syn.min()) / (syn.max() - syn.min()) - 1
    # impmax, impmin = logimp.max(), logimp.min()
    # logimp = (logimp - impmin) / (impmax - impmin)
    # implow = (implow - impmin) / (impmax - impmin)
    # s = 2.073179262737121

    patch_syn = DataProess(syn, args)
    patch_logimp = DataProess(logimp, args)
    patch_implow = DataProess(implow, args)
    patch_cfs = [DataProess(cfs[i, :, :], args) for i in range(cfs.shape[0])]
    patch_cfs = torch.cat(patch_cfs, dim=1)
    print(patch_syn.shape)
    patch_labels = np.expand_dims(get_patches(labels, args.patchsize, args.stride), 1)
    patch_labels = torch.tensor(patch_labels).to(device).type(torch.float32)

    patch_weight = np.expand_dims(get_patches(weight, args.patchsize, args.stride), 1)
    patch_weight = torch.tensor(patch_weight).to(device).type(torch.float32)

    patch_weight1 = np.expand_dims(get_patches(weight1, args.patchsize, args.stride), 1)
    patch_weight1 = torch.tensor(patch_weight1).to(device).type(torch.float32)

    patch_angle = np.expand_dims(get_patches(angle, args.patchsize, args.stride), 1)
    patch_angle = torch.tensor(patch_angle).to(device).type(torch.float32)
    # print(patch_syn.shape, patch_logimp.shape, patch_implow.shape, patch_labels.shape, patch_cfs.shape, patch_angle.shape)
    d = torch.cat([patch_syn, patch_cfs[:, -1, :, :].unsqueeze(1), patch_cfs[:, 0, :, :].unsqueeze(1)], dim=1)
    if not test:
        train_loader = DataLoader(
            TensorDataset(patch_syn, patch_logimp, patch_implow, patch_labels, patch_cfs, patch_angle, patch_weight, patch_weight1),
            batch_size=args.batch_size, shuffle=False)  # 保持数据原始顺序
        return train_loader, wav, cfs_wavelets, m_min_max[1] / s_min_max[1], m_min_max[1] / cfs_min_max[
            1], m_min_max[1] / high_min_max[1], logimp.shape
    else:
        test_loader = DataLoader(
            TensorDataset(patch_syn, patch_logimp, patch_implow, patch_cfs, patch_angle),
            batch_size=args.batch_size, shuffle=False, drop_last=False)  # 最佳num_workers=5、6
        return test_loader, wav, m_min_max, mlow_min_max, s_min_max, logimp.shape


# ----------------------------------------------------------------------------------------------------------------------
def get_models(args, drop_path_rate=0.1, dp=4, flag=0):
    initial_checkpoint = '/home/shendi_gjh_pq/workspace/IMP2d/checkpoints/Sep01_234515_trans_base_clean_epoch_200'
    pre_dict = torch.load(initial_checkpoint).state_dict()

    # net = MyNet.UNet2D(in_ch=2, out_ch=1, down_channels=16, layers=3, skip_channels=4, use_norm=False, use_att=True)
    net = STMNet_1(in_ch=2, out_ch=1, input_size=args.patchsize[0], embed_dim=16, drop_path_rate=drop_path_rate,
                   win_size=4, dp=dp)
    # net = CLWTNet(in_ch=2, input_size=args.patchsize[0], patch_size=8, head=4, depth=1)

    if args.use_pretrained:
        print('Using pretrained model')
        new_state_dict = OrderedDict()
        for k, v in pre_dict.items():
            name = k.replace('module.', '')  # 去掉 'module.' 前缀
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict)

    optimizer = optim.Adam(list(net.parameters()), amsgrad=True, lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch, eta_min=1e-06)
    if torch.cuda.is_available():
        net = nn.DataParallel(net, device_ids=[0, 1, 2])  # [1,3]
        net.to(device)
    return net, optimizer, scheduler


# ---------------------------------------------------------------------------------------------------------------------------

def loss_process(args, loss_fn, book, out, weight, weight1):
    if args.reg == 'tv':
        loss1 = loss_fn(conv_1D(book[-2], Pylop_diff(out), book[-1]), book[0])
        loss2 = tv_loss(out, weight, weight1)
        loss3 = loss_fn(book[3] * out, book[3] * book[1])
        loss = args.alfa * loss1 + loss2 + args.yita * loss3
    elif args.reg == 'dip':
        loss1 = loss_fn(conv_1D(book[-2], Pylop_diff(out), book[-1]), book[0])
        loss2 = dip_loss(out, book[-3], args, weight, weight1)
        loss3 = loss_fn(book[3] * out, book[3] * book[1])
        loss = args.alfa * loss1 + loss2 + args.yita * loss3
    else:
        loss1 = loss_fn(conv_1D(book[-2], Pylop_diff(out), book[-1]), book[0])
        loss2 = torch.tensor(0.0).type(torch.float32)
        loss3 = loss_fn(book[3] * out, book[3] * book[1])
        loss = args.alfa * loss1 + args.yita * loss3
    return loss, loss1, loss2, loss3


# ----------------------------------------------------------------------------------------------------------------------
def train(args):  # 0.01 1
    train_loader, wav, wavelets, s, s0, s1, impsize = get_data(args)
    nets = []
    optimizers = []
    schedulers = []
    if args.reg == 'tv':
        print('tv')
    elif args.reg == 'dip':
        print('dip')
    else:
        print('base')
    for i in range(args.iters):
        net, optimizer, scheduler = get_models(args)
        print(f'iter{i + 1}')
        if i == 1:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * 1
        nets.append(net)
        optimizers.append(optimizer)
        schedulers.append(scheduler)
    if args.stage == 2:
        print('stage2')
        first_step_model = torch.load("checkpoints/Aug23_235748_test_stage1")
        nets[-1].load_state_dict(first_step_model.state_dict())
    for net in nets:
        net.train()
    loss_fn = nn.MSELoss().to(device)
    args.cfs_weights = torch.tensor(args.cfs_weights).to(device).type(torch.float32)
    if not isdir("checkpoints"):
        os.mkdir("checkpoints")
    print("Training the model")
    train_loss = []
    predict_label = []
    train_psnr = []
    for epoch in tqdm(range(args.max_epoch), colour='green'):
        batch_loss, loss1_sum, loss2_sum, loss3_sum, ba_psnr, ba_psnr0, ba_psnr1, loss1_sum_2, loss3_sum_2 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        for i, (d, m, mlow, labels, cfs, angle, weight, weight1) in enumerate(train_loader, 0):
            book = [d, m, mlow, labels, angle, wav, s]
            out_iters = []
            losses, losses1, losses2, losses3 = [], [], [], []
            out = mlow
            # Iterative Gradient Corrected
            for k in range(args.iters):
                out = nets[k](torch.cat((d, out), dim=1)) + out
                out_iters.append(out)

            for out_iter in out_iters:
                loss_iter, loss1_iter, loss2_iter, loss3_iter = loss_process(args, loss_fn, book, out_iter, weight, weight1)
                losses.append(loss_iter)
                losses1.append(loss1_iter)
                losses2.append(loss2_iter)
                losses3.append(loss3_iter)
            if args.iters == 1:
                loss = losses[-1]
            else:
                loss = 0.2 * losses[0] + 0.8 * losses[-1]
            loss1 = losses1[-1]
            loss2 = losses2[-1]
            loss3 = losses3[-1]

            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(nets[-1].parameters(), max_norm=1)
            for optimizer in optimizers:
                optimizer.step()
                optimizer.zero_grad()
            batch_loss += loss.item()
            loss1_sum += (args.alfa * loss1).item()
            loss1_sum_2 += (args.alfa * losses1[0]).item()
            # loss2_sum += (args.mu * loss2).item() if args.reg == 'tv' else loss2.item() if args.reg == 'dip' else 0
            loss2_sum += loss2.item() if args.reg == 'tv' else loss2.item() if args.reg == 'dip' else 0
            loss3_sum += (args.yita * loss3).item()
            loss3_sum_2 += (args.yita * losses3[0]).item()
            ba_psnr += batch_snr(out_iters[-1][:, 0, :, :].unsqueeze(1).detach(), m.detach())
            if args.iters > 1:
                ba_psnr1 += batch_snr(out_iters[0][:, 0, :, :].unsqueeze(1).detach(), m.detach())
        for scheduler in schedulers:
            scheduler.step()
        # schedulers[-1].step()
        epoch_loss = batch_loss / (i + 1)
        train_loss.append(epoch_loss)
        epoch_loss1 = loss1_sum / (i + 1)
        epoch_loss2 = loss2_sum / (i + 1)
        epoch_loss3 = loss3_sum / (i + 1)
        epoch_loss1_2 = loss1_sum_2 / (i + 1)
        epoch_loss3_2 = loss3_sum_2 / (i + 1)
        epoch_psnr = ba_psnr / (i + 1)
        epoch_psnr1 = ba_psnr1 / (i + 1)
        train_psnr.append(epoch_psnr)
        current_lr = optimizers[-1].param_groups[0]['lr']
        print(
            f"  loss: {epoch_loss:.5f}, data_misfit: {epoch_loss1:.5f}, data_misfit_iter1: {epoch_loss1_2:.5f},"
            f" reg_loss: {epoch_loss2:.5f},"
            f" label_loss: {epoch_loss3:.5f}, label_loss_iter1: {epoch_loss3_2:.5f}"
            f" lr: {current_lr:.6f}, psnr_iter1: {epoch_psnr1:.3f}, psnr_iter2: {epoch_psnr:.3f}")
        if (epoch + 1) % args.gap == 0:
            for i in range(args.iters):
                torch.save(nets[i], f"checkpoints/{args.session_name}_{args.netname}_epoch_{epoch + 1}_iter{i + 1}")
    return train_loss, train_psnr, predict_label


# ----------------------------------------------------------------------------------------------------------------------
def process_predict(data, args, data_size, data_max):
    predicted_data = ((torch.cat(data, dim=0)).cpu().numpy())
    predicted_data = back_image(predicted_data.squeeze(), data_size, args.stride)
    predicted_data = denor_data(predicted_data, data_max)
    return predicted_data


# ----------------------------------------------------------------------------------------------------------------------
def test(args):
    test_loader, wav, m_min_max, mlow_min_max, s_min_max, impsize = get_data(args, test=True)
    s = m_min_max[1] / s_min_max[1]
    epochs = list(range(args.gap, args.max_epoch + 1, args.gap))
    predicted_imp_all = np.zeros((len(epochs), *impsize))  # Initialize 3D array
    predicted_imp_all0 = np.zeros((len(epochs), *impsize))  # 用来储存第一次迭代的结果 当迭代两次时
    for idx, epoch in enumerate(epochs):
        checkpoint_paths = []
        for i in range(args.iters):
            checkpoint_paths.append(f"checkpoints/{args.session_name}_{args.netname}_epoch_{epoch}_iter{i + 1}")
            print(f'iter {i + 1} times')
        if not os.path.exists(checkpoint_paths[0]):
            continue
        nets = []
        for checkpoint_path in checkpoint_paths:
            nets.append(torch.load(checkpoint_path))
        loss_fn = nn.MSELoss().to(device)
        predicted_imp = []
        predicted_imp0 = [] if args.iters > 1 else None
        predicted_st = []
        predicted_pool = []
        predicted_up = []
        true_imp = []
        true_low = []
        test_loss = []
        for net in nets:
            net.eval()
        print(f"\nTesting the model at epoch {epoch}")
        with torch.no_grad():
            # for d, m, mlow, cfs, angle in test_loader:
            for d, m, mlow, cfs, angle in tqdm(test_loader, colour='blue'):
                out_iters = []
                out_iters_st = []
                out_iters_pool = []
                out_iters_up = []
                out = mlow
                for k in range(args.iters):
                    out = nets[k](torch.cat((d, out), dim=1)) + out
                    out_iters.append(out)

                loss = loss_fn(out_iters[-1], m)
                test_loss.append(loss.item())
                true_imp.append(m)
                true_low.append(mlow)
                predicted_imp.append(out_iters[-1])
                if args.iters > 1:
                    predicted_imp0.append(out_iters[0])
        predicted_imp = process_predict(predicted_imp, args, impsize, m_min_max)
        if args.iters > 1:
            predicted_imp0 = process_predict(predicted_imp0, args, impsize, m_min_max)
        true_imp = process_predict(true_imp, args, impsize, m_min_max)
        true_low = process_predict(true_low, args, impsize, mlow_min_max)
        predicted_imp_all[idx, :, :] = predicted_imp  # Store in 3D array
        if args.iters > 1:
            predicted_imp_all0[idx, :, :] = predicted_imp0
        true_syn = forwardblock(true_imp, f0=30, dt=0.001, wave_len=51).cleansyn()
        reconstructed_syn = forwardblock(predicted_imp, f0=30, dt=0.001, wave_len=51).cleansyn()
        if args.data_flag == 'model':
            loss1, corr1, psnr1, ssim1 = evaluation(predicted_imp, true_imp)
            loss2, corr2, psnr2, ssim2 = evaluation(reconstructed_syn, true_syn)
            print(f"Iter -1:\n"
                  f"Epoch {epoch} - AI : MSE: {loss1:.5f}, Corr: {corr1:.5f}, psnr: {psnr1:.5f}, ssim: {ssim1:.5f}\n"
                  f"Epoch {epoch} - Syn: MSE: {loss2:.5f}, Corr: {corr2:.5f}, psnr: {psnr2:.5f}, ssim: {ssim2:.5f}")
            if args.iters > 1:
                loss1, corr1, psnr1, ssim1 = evaluation(predicted_imp0, true_imp)
                print(f"Iter 0:\n"
                      f"Iter2 Epoch {epoch} - AI : MSE: {loss1:.5f}, Corr: {corr1:.5f}, psnr: {psnr1:.5f}, ssim: {ssim1:.5f}")
        else:
            loss1, corr1, psnr1, ssim1 = evaluation(predicted_imp[:, 53], true_imp[:, 53])
            print(f"Epoch {epoch} - AI : MSE: {loss1:.5f}, Corr: {corr1:.5f}, psnr: {psnr1:.5f}, ssim: {ssim1:.5f}")
    return predicted_imp_all, true_imp, true_low, predicted_imp_all0

