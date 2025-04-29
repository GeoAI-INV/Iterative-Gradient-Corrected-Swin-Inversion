import numpy as np
from scipy.signal import filtfilt
from scipy import signal
import math
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from scipy.ndimage import gaussian_filter


# ----------------------------------------------------------------------------------------------------------------------
def add_gaussian_noise(image, snr_db, mode='gaussian'):
    image_array = np.array(image)
    signal_power = np.mean(image_array ** 2)

    snr = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr
    base_noise = np.random.normal(scale=np.sqrt(noise_power), size=image_array.shape)
    processed_noise = base_noise.copy()
    if mode == 'bandlimited':
        B, A = signal.butter(2, 0.2, 'low')  # 2*f_target/f_s remove frequency below f_target 2 * 6 / 1000  fs=1/dt 0.012
        processed_noise = signal.filtfilt(B, A, processed_noise.T).T
        print('bandlimited noise')
    elif mode == 'coherent':
        processed_noise = gaussian_filter(processed_noise, sigma=3)
        print('coherent noise')
    current_power = np.mean(processed_noise ** 2)
    if current_power > 0:
        scale = np.sqrt(noise_power / current_power)
        processed_noise *= scale

    noisy_image = image_array + processed_noise

    return noisy_image


def single_snr(signal, noise_data):
    signal = np.array(signal)
    noise_data = np.array(noise_data)
    p_signal = np.mean((signal) ** 2)
    p_noise = np.mean((noise_data - signal) ** 2)
    if p_noise == 0:
        return 999
    snr = 10 * math.log10(p_signal / p_noise)
    return snr


def batch_snr(pre_data, clean_data):
    Pre_data = pre_data.cpu().numpy()  # detach().
    Clean_data = clean_data.data.cpu().numpy()
    SNR = 0
    for i in range(Pre_data.shape[0]):
        Pre = Pre_data[i, :, :, :].squeeze()  # 默认压缩所有为1的维度
        Clean = Clean_data[i, :, :, :].squeeze()
        SNR += psnr(Pre, Clean, data_range=Clean.max() - Clean.min())
    return SNR / Pre_data.shape[0]


# ----------------------------------------------------------------------------------------------------------------------
def denor_data(data, tn):
    return data * tn[1]


# ----------------------------------------------------------------------------------------------------------------------
def evaluation(predict, true):
    loss = mean_squared_error(true, predict)
    corr = pearsonr(true.flatten(), predict.flatten())[0]
    psnr_ = psnr(true, predict, data_range=true.max() - true.min())
    ssim = compare_ssim(true, predict, win_size=21, data_range=true.max() - true.min())
    return loss, corr, psnr_, ssim


# ---------------------------------------------------------------------------------------------------------------------
def get_patches(image, patch_size, stride):
    """
        image:需要切分为图像块的图像       
        patch_size:图像块的尺寸，如:(10,10)        
        stride:切分图像块时移动过得步长，如:5        
        """

    if len(image.shape) == 2:
        # 灰度图像        
        imhigh, imwidth = image.shape
    if len(image.shape) == 3:
        # RGB图像       
        imhigh, imwidth, imch = image.shape

    # 构建图像块的索引
    if imhigh == patch_size[0]:
        range_y = [0]
    else:
        range_y = np.arange(0, imhigh - patch_size[0], stride[0])
    range_x = np.arange(0, imwidth - patch_size[1], stride[1])

    #  avoid forgetting the last row or col
    if range_y[-1] != imhigh - patch_size[0]:
        range_y = np.append(range_y, imhigh - patch_size[0])
    if range_x[-1] != imwidth - patch_size[1]:
        range_x = np.append(range_x, imwidth - patch_size[1])

    sz = len(range_y) * len(range_x)  # 图像块的数量

    if len(image.shape) == 2:
        # 初始化灰度图像
        res = np.zeros((sz, patch_size[0], patch_size[1]))
    if len(image.shape) == 3:
        # 初始化RGB图像
        res = np.zeros((sz, patch_size[0], patch_size[1], imch))

    index = 0
    for y in range_y:
        for x in range_x:
            res[index] = image[y:y + patch_size[0], x:x + patch_size[1]]
            index = index + 1

    return res


# ----------------------------------------------------------------------------------------------------------------------
def back_image(patches, imsize, stride):
    """
        patches: 使用get_patches得到的数据        
        imsize:原始图像的宽和高，如(321, 481)       
        stride:图像切分时的步长，如10        
        """
    patch_size = patches.shape[-2:]  #[-2:]
    if len(patches.shape) == 3:
        ## 初始化灰度图像       
        res = np.zeros((imsize[0], imsize[1]))
        w = np.zeros(((imsize[0], imsize[1])))
    if len(patches.shape) == 4:
        ## 初始化RGB图像        
        res = np.zeros((imsize[0], imsize[1], 3))
        w = np.zeros(((imsize[0], imsize[1], 3)))

    if imsize[0] == patch_size[0]:
        range_y = [0]
    else:
        range_y = np.arange(0, imsize[0] - patch_size[0], stride[0])
    range_x = np.arange(0, imsize[1] - patch_size[1], stride[1])

    if range_y[-1] != imsize[0] - patch_size[0]:
        range_y = np.append(range_y, imsize[0] - patch_size[0])
    if range_x[-1] != imsize[1] - patch_size[1]:
        range_x = np.append(range_x, imsize[1] - patch_size[1])

    index = 0
    for y in range_y:
        for x in range_x:
            res[y:y + patch_size[0], x:x + patch_size[1]] = res[y:y + patch_size[0], x:x + patch_size[1]] + patches[
                index]
            w[y:y + patch_size[0], x:x + patch_size[1]] = w[y:y + patch_size[0],
                                                          x:x + patch_size[1]] + 1  #每个位置的重复的权重 最后除以重复的次数
            index = index + 1

    return res / w


# ----------------------------------------------------------------------------------------------------------------------
def get_low(data, x_scale, z_scale):
    B, A = signal.butter(2, 0.012, 'low')  # 2*f_target/f_s remove frequency below f_target 2 * 6 / 1000  fs=1/dt 0.012
    m_loww = signal.filtfilt(B, A, data.T).T
    nsmooth = x_scale
    m_low = filtfilt(np.ones(nsmooth) / float(nsmooth), 1,
                     m_loww)  # small smooth along with the x axis and save some information
    nsmooth = z_scale
    m_low = filtfilt(np.ones(nsmooth) / float(nsmooth), 1, m_low.T).T  # big smooth along with the z axis
    return m_low

