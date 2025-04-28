"@author: PangQi"
import numpy as np
import pylops
from pylops.utils.wavelets import ricker
from utils.functions import add_gaussian_noise
import pywt


class forwardblock():
    def __init__(self, logimp, f0, dt, wave_len):
        super().__init__()
        "logimp log impedance"
        "wave_len: half of wavelet len and including zero point 1+50"
        "noise_sd: the standard of noise"
        self.logimp = logimp
        self.f0 = f0
        self.dt = dt
        self.wave_len = wave_len

    def wavelet(self):
        wav, twav, wavc = ricker(np.arange(self.wave_len) * self.dt, self.f0)
        return wav

    def mor_wavelet(self):
        morlet_wavelet = pywt.ContinuousWavelet('cmor0.5-1')
        morlet_wavelet.center_frequency = 0.7
        morlet_wavelet.bandwidth_frequency = 0.34
        morl_wave = morlet_wavelet.wavefun(length=self.wave_len * 2 - 1)[0]
        return np.real(morl_wave)

    def cleansyn(self):
        wav = self.wavelet()
        dims = self.logimp.shape
        PPop = pylops.avo.poststack.PoststackLinearModelling(wav, nt0=dims[0], spatdims=dims[1])
        return PPop * self.logimp

    def noisesyn(self, snr_db, mode='gaussian'):
        return add_gaussian_noise(self.cleansyn(), snr_db, mode=mode)

    def morl_syn(self):
        wav = self.mor_wavelet()
        dims = self.logimp.shape
        PPop = pylops.avo.poststack.PoststackLinearModelling(wav, nt0=dims[0], spatdims=dims[1])
        return PPop * self.logimp


class Avoforwardblock():
    def __init__(self, m, f0, dt, wave_len, snr_db):
        super().__init__()
        "wave_len: half of wavelet len and including zero point 1+50"
        "noise_sd: the standard of noise"
        self.m = m  # log(vp,vs,den)
        self.f0 = f0
        self.dt = dt
        self.wave_len = wave_len
        self.snr_db = snr_db
        self.vsvp = 0.5
        self.ntheta = 3
        thetamin, thetamax = 5, 25  # 不能超过33° 超过后相位反转
        self.theta = np.linspace(thetamin, thetamax, self.ntheta)

    def wavelet(self):
        wav, twav, wavc = ricker(np.arange(self.wave_len) * self.dt, self.f0)
        return wav

    def cleanag(self):
        wav = self.wavelet()
        dims = self.m.shape
        PPop_dense = pylops.avo.prestack.PrestackLinearModelling(
            wav,
            self.theta,
            vsvp=self.vsvp,
            nt0=dims[0],
            spatdims=(dims[2],),
            linearization="akirich",
            explicit=True,
        )
        dPP = PPop_dense * self.m.swapaxes(0, 1).ravel()
        dPP = dPP.reshape(self.ntheta, dims[0], dims[2]).swapaxes(0, 1)
        return dPP

    def noiseag(self):
        noise_data = 0 * self.cleanag()
        for i in range(self.m.shape[1]):
            noise_data[:, i, :] = add_gaussian_noise(self.cleanag()[:, i, :], self.snr_db)
        return noise_data


if __name__ == '__main__':
    from scipy.io import loadmat
    import matplotlib.pyplot as plt
    import torch
    import torch.nn.functional as F

    from utils.functions import get_patches, back_image

    plt.rcParams.update({
        "font.family": 'serif',
        "mathtext.fontset": 'stix',
        "font.serif": ['Times New Roman'],
        'xtick.labelsize': 18,  # x轴刻度标签字体大小
        'ytick.labelsize': 18,  # y轴刻度标签字体大小
        'axes.labelsize': 24,  # 坐标轴标签字体大小
        'legend.fontsize': 18,  # 图例字体大小
        'figure.dpi': 400,  # 分辨率
        'figure.facecolor': 'w',  # 背景颜色
    })
    patch_size = [256, 256]
    stride = [30, 30]

    vpk = loadmat('..//Data//vp.mat')['vp'][500:500 + 551, 0:13601:10]
    denk = loadmat('..//Data//den.mat')['den'][500:500 + 551, 0:13601:10]
    logimp = np.log(vpk * denk)
    synblock = forwardblock(logimp, f0=30, dt=0.001, wave_len=51)
    syn = synblock.cleansyn()
    patch_syn = np.expand_dims(get_patches(syn, patch_size, stride), 1)
    
    patch_imp = np.expand_dims(get_patches(logimp, patch_size, stride), 1)
    
    plt.figure(figsize=(6, 6), layout='constrained')
    plt.imshow(patch_imp[-39, 0, :, :], 'jet')
    plt.show()
  
            
    plt.figure(figsize=(8, 8))
    plt.imshow(syn[  0:500, -500:], 'seismic', vmin=-0.1, vmax=0.1)
    plt.show()

    # syn1 = synblock.noisesyn(snr_db=10)
    # syn2 = synblock.noisesyn(snr_db=10, mode='bandlimited')
    # syn3 = synblock.noisesyn(snr_db=10, mode='coherent')

    # plt.show()
    # fig, ax = plt.subplots(figsize=(10, 6))
    # ax.plot(syn[:, 20], label='clean')
    # ax.plot(syn1[:, 20], label='gau')
    # ax.plot(syn2[:, 20], label='band')
    # ax.plot(syn3[:, 20], label='coh')
    # ax.legend()
    # ax.set_title('Single Trace ma')
    # plt.tight_layout()  # 自适应调整布局

    # fig, ax = plt.subplots(figsize=(10, 6))
    # ax.plot(patch_syn[5,0,:,3], label='real')
    # ax.plot(pytorch_conv[5,0,:,3], label='pytorch')
    # ax.legend()
    # ax.set_title('Single Trace ma')
    # plt.tight_layout()  # 自适应调整布局
    # plt.show()

    # wav=synblock.wavelet()
    # nt0=logimp.shape[0]
    # syn=synblock.cleansyn()

    # patch_syn = np.expand_dims(get_patches(syn, patch_size, stride), 1)

    # patch_imp = np.expand_dims(get_patches(logimp, patch_size, stride), 1)

    # def DIFFZ(z):
    #     DZ = torch.zeros([z.shape[0], z.shape[1], z.shape[2], z.shape[3]])
    #     DZ[..., 1:-1,:] = 0.5 * (z[..., 2:,:] - z[..., :-2,:])
    #     return DZ

    # def conv_1D(wav, r):
    #     WEIGHT = torch.tensor(wav.reshape(1, 1, wav.shape[0])).type(torch.float32)
    #     For_syn = 0 * r
    #     for i in range(r.shape[3]):
    #         INPUT = r[..., i]
    #         For_syn[..., i] = F.conv1d(INPUT, WEIGHT, stride=1, padding=(len(wav) - 1) // 2)
    #     return For_syn

    # # rr=torch.tensor(logimp).type(torch.float32).reshape(1,1,logimp.shape[0],logimp.shape[1])
    # pytorch_conv=conv_1D(wav,DIFFZ(torch.tensor(patch_imp).type(torch.float32))).numpy()
    # # err=pytorch_conv-syn
    # rec = back_image(pytorch_conv.squeeze(), logimp.shape, stride)
    # rec_real = back_image(patch_syn.squeeze(), logimp.shape, stride)

    # check=patch_imp.squeeze()
    # # check_3 = conv_1D(wav,DIFFZ(torch.tensor(patch_imp[2,0,:,:]).type(torch.float32))).numpy()

    # fig, ax = plt.subplots(figsize=(10, 6))
    # ax.plot(patch_syn[0,0,:,3], label='real')
    # ax.plot(pytorch_conv[0,0,:,3], label='pytorch')
    # ax.legend()
    # ax.set_title('Single Trace ma')
    # plt.tight_layout()  # 自适应调整布局

    # plt.show()
    # fig, ax = plt.subplots(figsize=(10, 6))
    # ax.plot(patch_syn[1,0,:,3], label='real')
    # ax.plot(pytorch_conv[1,0,:,3], label='pytorch')
    # ax.legend()
    # ax.set_title('Single Trace ma')
    # plt.tight_layout()  # 自适应调整布局

    # plt.show()
    # fig, ax = plt.subplots(figsize=(10, 6))
    # ax.plot(patch_syn[2,0,:,3], label='real')
    # ax.plot(pytorch_conv[2,0,:,3], label='pytorch')
    # ax.legend()
    # ax.set_title('Single Trace ma')
    # plt.tight_layout()  # 自适应调整布局
    # plt.show()

    # fig, ax = plt.subplots(figsize=(10, 6))
    # ax.plot(patch_syn[5,0,:,3], label='real')
    # ax.plot(pytorch_conv[5,0,:,3], label='pytorch')
    # ax.legend()
    # ax.set_title('Single Trace ma')
    # plt.tight_layout()  # 自适应调整布局
    # plt.show()

    # fig, ax = plt.subplots(figsize=(10, 6))
    # ax.plot(syn[:,3], label='real')
    # ax.plot(rec_real[:,3], label='pytorch')
    # ax.legend()
    # ax.set_title('Single Trace ma')
    # plt.tight_layout()  # 自适应调整布局
    # plt.show()

    # synblock_check = forwardblock(logimp[:112,:112], f0=30, dt=0.001, wave_len=51, noise_sd=0)
    # err=pytorch_conv[0,0,:,:]-synblock_check.cleansyn()
    # plt.figure(figsize=(10, 6))
    # plt.imshow(synblock_check.cleansyn(), 'seismic')
    # plt.tight_layout()  # 调整子图布局，减少空白区域
    # plt.title('Predict unet IMP')
    # plt.show()
    # plt.figure(figsize=(10, 6))
    # plt.imshow(pytorch_conv[0,0,:,:], 'seismic')
    # plt.tight_layout()  # 调整子图布局，减少空白区域
    # plt.title('Predict unet IMP')
    # plt.show()

    # fig, ax = plt.subplots(figsize=(10, 6))
    # ax.plot(patch_syn[0,0,:,111], label='real')
    # ax.plot(synblock_check.cleansyn()[:,111], label='pytorch')
    # ax.legend()
    # ax.set_title('Single Trace ma')
    # plt.tight_layout()  # 自适应调整布局
    # plt.show()
