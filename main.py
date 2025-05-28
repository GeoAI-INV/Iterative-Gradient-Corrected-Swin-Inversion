"""
Created on Jun 1 2024
@author: Pang Qi
"""
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1, 2, 3'  # it should be put in the beginning
from utils.basic_tools.basic_settings import set_deterministic_seed
from core.common import train, test

import argparse
from datetime import datetime
import time
from scipy.io import savemat

set_deterministic_seed(333)


def Save(data, data_name):
    savemat(f'{"/home/shendi_gjh_pq/workspace/IMP2d/data_paper"}/{data_name}.mat', {data_name: data})


def Save_loss(loss, psnr, data_name):
    savemat(f'{"/home/shendi_gjh_pq/workspace/IMP2d/data_paper/loss_psnr"}/{data_name}_loss.mat', {f'{data_name}_loss': loss})
    savemat(f'{"/home/shendi_gjh_pq/workspace/IMP2d/data_paper/loss_psnr"}/{data_name}_psnr.mat', {f'{data_name}_psnr': psnr})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-labels', type=int, default=[200, 500, 750, 1100],
                        help="Number of AI traces from the model to be used for training in marimous")
    parser.add_argument('-max_epoch', type=int, default=50, help="Maximum number of training epochs")
    parser.add_argument('-batch_size', type=int, default=16, help="Batch size for training")
    parser.add_argument('-wave', type=str, default='haar', help="Wavelet type to be used (e.g., haar, morlet, etc.)")
    parser.add_argument('-reg', type=str, default='base', help="Regularization methods ('base':without reg. 'tv':with tv reg. 'dip':with aso reg)")
    parser.add_argument('-patchsize', type=int, default=[112, 112], help="Size of each patch for input data [height, width]")
    parser.add_argument('-stride', type=int, default=[20, 5], help="Stride for patch extraction [height_stride, width_stride]")
    parser.add_argument('-Nt', type=int, default=551, help="Time samples")
    parser.add_argument('-lr', type=float, default=0.01, help="Learning rate for training")
    parser.add_argument('-test_checkpoint', type=str, default=None, help="Path to the model for testing without training")
    parser.add_argument('-pretest_checkpoint', type=str, default=None, help="Path to the model for pre-testing without training")
    parser.add_argument('-session_name', type=str, default=datetime.now().strftime('%b%d_%H%M%S'), help="Name of the session for saving the model")
    parser.add_argument('-netname', type=str, default='sw-g-bgp', help="Name for saving the model")
    parser.add_argument('-alfa', type=float, default=1, help="Data misfit weight")
    parser.add_argument('-mu', type=float, default=0.05, help="Total variation (TV) loss weight in horizontal")
    parser.add_argument('-mu1', type=float, default=0.05, help="Total variation (TV) loss weight in vertical")
    parser.add_argument('-hor', type=float, default=0.9, help="Horizontal dip loss weight")
    parser.add_argument('-ver', type=float, default=0.001, help="Vertical dip loss weight")
    parser.add_argument('-yita', type=float, default=1, help="Label loss weight")
    parser.add_argument('-lamda', type=float, default=0, help="Low-frequency loss weight")
    parser.add_argument('-weights', type=str, default=[1, 1, 1.4], help="Weights for different AVO parameters")
    parser.add_argument('-noise_flag', type=str, default=None, help="Whether to use noisy data for training")
    parser.add_argument('-data_flag', type=str, default='model', help="Choose the data type (e.g., model, field)")
    parser.add_argument('-stage', type=int, default=1, help="Training stage: 1 for full training, 2 for loading pre-trained model")
    parser.add_argument('-angle', type=str, default='direct', help="Method to calculate dip angle")
    parser.add_argument('-use_noise', type=str, default=None, help="Whether to use noise in the data")
    parser.add_argument('-snr', type=int, default=10, help="Signal-to-noise ratio (SNR) for the data")
    parser.add_argument('-cfs_weights', type=str, default=[1], help="CFS frequency weights")
    parser.add_argument('-use_pretrained', type=str, default=None, help="Whether to use pre-trained model")
    parser.add_argument('-align', type=str, default=None, help="Whether to use alignment for data")
    parser.add_argument('-iters', type=int, default=1, help="Number of iterations for the training process")
    parser.add_argument('-gap', type=int, default=50, help="Number of epochs between model checkpoints")
    parser.add_argument('-net_mode', type=str, default=None, help="Network type to use in the first stage")
    parser.add_argument('-noise_type', type=str, default='gaussian', help="Type of noise (e.g., gaussian, uniform)")
    parser.add_argument('-seam', type=str, default=None, help="Whether to use the SEAM model")
    args = parser.parse_args()

    # # ---------------------------------------------------------------Train and inference
    # Parameters：
    # 1. marimous model：
    #    - labels: [200, 500, 750, 1100] - batch size: 32 in ustnet - patch size: [256, 256] - stride: [30, 30] - get_low: [400, 180] - win_size: 8
    #    - learning rate: 1e-3~1e-6 - max epoch: 200
    # 2. seam model：
    #    - labels: [150, 300, 450, 600, 750] - batch size: 32 in ustnet - patch size: [256, 256] - stride: [30, 30] - get_low: [100, 50] - win_size: 8
    #    - learning rate: 1e-3 or 1e-4~1e-5 - max epoch: 200
    # below is an example of how to set the parameters for training and testing the model.
    args.data_flag = 'model'
    args.align = True
    args.labels = [150, 300, 450, 600, 750]
    args.batch_size = 32
    args.stage = 1
    args.lr = 1e-4
    args.max_epoch = 200
    args.gap = 100
    args.patchsize = [256, 256]
    args.stride = [30, 30]
    args.alfa = 0.01
    args.yita = 1000
    args.mu = 0.3
    args.mu1 = 0.5
    args.hor = 0.4
    args.ver = 0.001
    args.use_pretrained = None
    args.reg = 'dip'
    args.angle = 'pwd'
    args.iters = 2
    # args.use_noise = True
    # args.noise_type = 'bandlimited'
    # args.seam = True
    args.snr = 15
    # save_name = "noisebandlimited_10db_trans_iter1_dipbig_32"  
    save_name = "test_mar"
    # args.session_name = "field_aso"
    args.netname = save_name
    st_time = time.time()
    train_loss, train_psnr, predict_label = train(args)
    ed_time = time.time()
    total_time = ed_time - st_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Total time: {int(hours)}h{int(minutes)}min{seconds:.2f}s")
    predicted_imp, true_imp, ture_low, predicted_imp0 = test(args)  # the last iter->predicted_imp
    print(predicted_imp.shape)

    Save(predicted_imp, save_name)
    Save_loss(train_loss, train_psnr, save_name)
