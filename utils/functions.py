import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.signal import filtfilt
from scipy import signal
import torch.nn.functional as F
import math
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from scipy.ndimage import gaussian_filter
# ----------------------------------------------------------------------------------------------------------------------
def add_gaussian_noise(image, snr_db, mode = 'gaussian'):
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


# def psnr(img1, img2):
#     mse = np.mean((img1 - img2) ** 2)
#     if mse == 0:
#         return 999
#     PIXEL_MAX = 255.0
#     return 10 * math.log10(PIXEL_MAX * PIXEL_MAX / mse)


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
def normalize_data(data):
    mean_val = np.mean(data)
    std_val = np.std(data)
    normalized_data = (data - mean_val) / std_val
    return normalized_data

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


# ----------------------------------------------------------------------------------------------------------------------
def DIFFZ(z):
    DZ = torch.zeros([z.shape[0], z.shape[1], z.shape[2], z.shape[3]]).type(torch.float32)
    DZ[..., :-1, :] = 0.5 * (z[..., 1:, :] - z[..., :-1, :])
    return DZ


# ----------------------------------------------------------------------------------------------------------------------
# def tv_loss(x):
#     dh = torch.abs(x[..., :, 1:] - x[..., :, :-1])
#     dw = torch.abs(x[..., 1:, :] - x[..., :-1, :])
#     return torch.mean(dh[..., :-1, :] + dw[..., :, :-1])

def tv_loss(x, weight):
    # Compute differences
    dw = torch.abs(torch.gradient(x, dim=3)[0])
    dh = torch.abs(torch.gradient(x, dim=2)[0])
    # Combine and average
    return torch.mean(weight*dw) + torch.mean(weight*dh)

# ----------------------------------------------------------------------------------------------------------------------
def L1_loss(x, beta):
    """
    L1-norm to add sparsity constraint
    """
    dw = torch.abs(x)
    return beta * torch.mean(dw)


# ----------------------------------------------------------------------------------------------------------------------
def make_yushi_wavelet(nYushiFreqLow, nYushiFreqHigh, nWaveletSample, dt):
    #p_par yushi 子波积分频率下界
    #q_par  子波积分频率上界
    #nsample：输出子波长度
    #s_interval:采样间隔
    p_par = nYushiFreqLow;
    q_par = nYushiFreqHigh;
    nsample = nWaveletSample;

    s_int = dt
    t = s_int * np.linspace(-(nsample // 2), nsample // 2, nsample);
    y = 1.0 / (q_par - p_par) * (q_par * np.exp(-np.power(3.1415926 * q_par * t, 2)) - p_par * np.exp(
        -np.power(3.1415926 * p_par * t, 2)));
    taper = np.ones([1, nsample], dtype='float32');
    taper[0, 0:nsample // 3] = 0.5 * (1 - np.cos(3.1415926 * np.arange(0, nsample // 3, 1) / (nsample // 3 - 1)))
    taper[0, nsample - nsample // 3:nsample] = 0.5 * (
            1 + np.cos(3.1415926 * np.arange(0, nsample // 3, 1) / (nsample // 3 - 1)))
    y_taper = np.multiply(y, taper);
    w = np.reshape(y_taper, (-1));
    return w;


# ----------------------------------------------------------------------------------------------------------------------
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

    ## 构建图像块的索引
    if imhigh == patch_size[0]:
        range_y = [0]
    else:
        range_y = np.arange(0, imhigh - patch_size[0], stride[0])
    range_x = np.arange(0, imwidth - patch_size[1], stride[1])

    #avoid forgetting the last row or col
    if range_y[-1] != imhigh - patch_size[0]:
        range_y = np.append(range_y, imhigh - patch_size[0])
    if range_x[-1] != imwidth - patch_size[1]:
        range_x = np.append(range_x, imwidth - patch_size[1])

    sz = len(range_y) * len(range_x)  ## 图像块的数量

    if len(image.shape) == 2:
        ## 初始化灰度图像        
        res = np.zeros((sz, patch_size[0], patch_size[1]))
    if len(image.shape) == 3:
        ## 初始化RGB图像       
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


# ----------------------------------------------------------------------------------------------------------------------
#路径回溯得到最优路径
def backtrack(R):
    path = np.zeros((2, 1))
    p, q = [], []
    i, j = np.array(R.shape) - 1
    while i > 0 or j > 0:
        p.append(i)
        q.append(j)
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            min_neighbor = min(R[i - 1][j], R[i][j - 1], R[i - 1][j - 1])
            if min_neighbor == R[i - 1][j]:
                i -= 1
            elif min_neighbor == R[i][j - 1]:
                j -= 1
            else:
                i -= 1
                j -= 1
    p.append(0), q.append(0)
    p.reverse(), q.reverse()
    path = np.pad(path, ((0, 0), (0, len(p) - path.shape[1])), mode='constant')
    path[0, :] = np.array(p)
    path[1, :] = np.array(q)
    return path


# ----------------------------------------------------------------------------------------------------------------------
def WarpData(y, path):
    xp = path[0, :]
    yp = path[1, :]

    interp_func = interp1d(xp, yp, kind='linear')

    warping_index = interp_func(np.arange(xp.min(), xp.max() + 1)).astype(np.int64)
    warping_index[0] = yp.min()
    warping_data = y[:, warping_index.T]

    return warping_data


#扭曲数据
# def WarpData(x,y,path): 
#     path=path.astype(int)
#     data_warpped=0*x
#     for i in range(x.shape[1]):
#         index=np.array(np.where(path[0] == i))
#         if index.shape[1]==1:
#             data_warpped[:,i]=y[:,path[1,index[0]]]
#         else:
#             data_warpped[:,i]=(np.sum(y[:,path[1,index[0,0]]:path[1,index[0,0]+index.shape[1]-1]+1]))/index.shape[1]
#             # data_warpped[:,i]=np.mean(y[:,path[1,index[0,0]]:path[1,index[0,0]+index.shape[1]]])
#     return data_warpped

# ----------------------------------------------------------------------------------------------------------------------
#生成最终扭曲数据
def WarpOut(y, path):
    # y=normal.unnormalize(y)
    #x为基准数据 y为待校准数据    
    if torch.is_tensor(y):
        if y.is_cuda:
            y = y.cpu()
        y = y.numpy()
    if torch.is_tensor(path):
        if path.is_cuda:
            path = path.cpu()
        path = path.numpy()
    out = y * 0
    for i in range(y.shape[0]):
        op_path = backtrack(path[i, :, :])
        out[i, :, :] = WarpData(y[i, :, :], op_path)  #转置是为了变成横向量

    out = torch.tensor(out).float()
    if torch.cuda.is_available():
        out = out.cuda()

    # out,nor_func=TorchData(out)

    return out


# ----------------------------------------------------------------------------------------------------------------------
def display_results(loss, property_corr, property_r2, args, header):
    property_corr = torch.mean(torch.cat(property_corr), dim=0).squeeze()  #dim=0按顺序拼接张量
    property_r2 = torch.mean(torch.cat(property_r2), dim=0).squeeze()
    loss = torch.mean(torch.tensor(loss))
    print("loss: {:.4f}\nCorrelation: {:0.4f}\nr2 Coeff.  : {:0.4f}".format(loss, property_corr, property_r2))


# def DataProess(data,args):
#     #standardization
#     data_stand =standardization(data)
#     #标准化
#     data = data_stand.standard(data)
#     #get patches
#     patch_data=np.expand_dims(get_patches(data,args.patchsize, args.stride),1)

#     patch_data = torch.tensor(patch_data, dtype=torch.float32).cuda()
#     return patch_data,data_stand

# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    import torch

    # 示例数据
    data = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    data = data.reshape(1, 1, -1)
    mean_val = torch.tensor(np.mean(data, keepdims=True)).float()
    std_val = torch.tensor(np.std(data, keepdims=True)).float()
    # 创建Normalization实例并设置均值和标准差
    normalizer = Normalization(mean_val, std_val)
    data = torch.tensor(data).float()

    # 对数据进行归一化
    normalized_data = normalizer.normalize(data)
    b = normalized_data.cpu().numpy()

    # 输出归一化后的数据
    print("Normalized Data:", b)

    # 对数据进行反归一化
    unnormalized_data = normalizer.unnormalize(normalized_data)
    c = unnormalized_data.cpu().numpy()
    # 输出反归一化后的数据
    print("Unnormalized Data:", c)
