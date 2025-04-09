
import pycuda.autoinit
from pycuda import gpuarray, compiler
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.driver as cuda
import tigre
import warnings
import luigi
import sys
import os
import shutil
import pickle
import logging
import astra
import time
import scipy.signal
import multiprocessing as mp
import gc
import h5py
import imageio.v3 as iio
import numpy as np
import tomopy
from tomopy.recon.rotation import (find_center_vo, find_center, find_center_pc)
import math
from rec_config import *
from bgcorrect2 import *
from wrapphase import *
import copy
from scipy.fft  import fft, ifft
from scipy.ndimage import map_coordinates
from scipy.io import loadmat
from scipy.signal import convolve
import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D


def filtering_single_angle(proj, geo, verbose=False):
    # 生成滤波器核（与角度无关）
    filt_len = max(64, 2**nextpow2(2*max(geo.nDetector)))
    ramp_kernel = ramp_flat(filt_len)
    
    # 应用滤波器窗函数
    d = 1  # 采样间隔
    filt = filter(geo.filter, ramp_kernel[0], filt_len, d, verbose=verbose)
    filt = np.kron(np.ones((geo.nDetector[0],1), dtype=np.float32), filt)
    
    # 计算零填充
    padding = (filt_len - geo.nDetector[1]) // 2
    
    # 修改比例因子（移除角度相关项）
    scale_factor = (geo.DSD/geo.DSO) / (4 * geo.dDetector[0]) 
    
    # 初始化复数容器
    fproj = np.zeros((geo.nDetector[0], filt_len), dtype=np.complex64)
    fproj.real[:, padding:padding+geo.nDetector[1]] = proj[0]
    
    # 频域滤波
    fproj = np.fft.fft(fproj, axis=1)
    fproj *= filt
    fproj = np.fft.ifft(fproj, axis=1)
    
    # 提取实部并缩放
    proj_filtered = np.real(fproj[:, padding:padding+geo.nDetector[1]]) * scale_factor
    return proj_filtered[np.newaxis,:,:]  # 保持维度一致

def filter(filter, kernel, order, d, verbose=False):
    f_kernel = abs(np.fft.fft(kernel)) * 2

    filt = f_kernel[: int((order / 2) + 1)]
    w = 2 * np.pi * np.arange(len(filt)) / order

    if filter in {"ram_lak", None}:
        if filter is None and verbose:
            warnings.warn("no filter selected, using default ram_lak")
    elif filter == "shepp_logan":
        filt[1:] *= np.sin(w[1:] / (2 * d)) / (w[1:] / (2 * d))
    elif filter == "cosine":
        filt[1:] *= np.cos(w[1:] / (2 * d))
    elif filter == "hamming":
        filt[1:] *= 0.54 + 0.46 * np.cos(w[1:] / d)
    elif filter == "hann":
        filt[1:] *= (1 + np.cos(w[1:] / d)) / 2
    # elif filter == "None":  
    #     filt = 1
    else:
        raise ValueError("filter not recognized: " + str(filter))

    filt[w > np.pi * d] = 0
    filt = np.hstack((filt, filt[1:-1][::-1]))
    return filt.astype(np.float32)

def ramp_flat(n, verbose=False):
    nn = np.arange(-n / 2, n / 2)
    h = np.zeros(nn.shape, dtype=np.float32)
    h[int(n / 2)] = 1 / 4
    odd = nn % 2 == 1
    h[odd] = -1 / (np.pi * nn[odd]) ** 2
    return h, nn

def nextpow2(n):
    i = 1
    while (2 ** i) < n:
        i += 1
    return i

# FORMAT = "pickle"
FORMAT = "hdf5"

def load_data(filename):
    if FORMAT == "pickle":
        with open(filename, "rb") as data_in:
            data = pickle.load(data_in)
    elif FORMAT == "hdf5":
        with h5py.File(filename) as f:
            data = {}
            for k in f.keys():
                data[k] = np.array(f[k])

    return data

def save_data(filename, data):
    if FORMAT == "pickle":    
        with open(filename,"wb") as data_out:
            pickle.dump(data, data_out, protocol=4)
    elif FORMAT == "hdf5":
        with h5py.File(filename, "w") as f:
            for k in data.keys():
                f.create_dataset(k, data=data[k])

def dataset_core_path(path):
    """
    Take the path to a dataset and return the path to the measurement without the suffix.

    Parameters
    ----------
    path : str
    
    Returns
    -------
    name : str

    """
    path1, path2 = os.path.split(path)
    return os.path.join(path1, "_".join(path2.split("_")[:2]))


class TimeTaskMixin(object):
    '''
    A mixin that when added to a luigi task, will print out
    the tasks execution time to standard out, when the task is
    finished
    '''
    @luigi.Task.event_handler(luigi.Event.PROCESSING_TIME)
    def print_execution_time(self, processing_time):
        print('### PROCESSING TIME ###: ' + str(processing_time))


class LoadRawData(luigi.Task):
    """
    Load data using Jincheng's code
    """
    data_name = luigi.Parameter()
    parameters = load_rec_config(data_name)
    data_in_path  = parameters['raw_data_path']
    data_out_path = parameters['processed_data_path']
    resources = { "network" : 1 }
    
    def output(self):
        """
        Write the data into an easy to handle file
        """
        filename = os.path.join(self.data_out_path, f"{self.data_name}_raw.{FORMAT}")
        return luigi.LocalTarget(filename)
    
    def run(self):
        """
        Load and sort out the data
        """

        logging.debug('LoadRawData Task')#记录了任务的开始，通过日志输出一条调试信息，表示 LoadRawData 任务已经开始执行。
        Ndark = self.parameters['Ndark']
        Nangles_f = self.parameters['Nangles_f']
        Nangles = self.parameters['Nangles']
        Nsteps = self.parameters['Nsteps']
        Ndarkstep = self.parameters['Ndarkstep']
        Nstages = self.parameters['Nstages']
        # 路径与名称参数，需要修改的部分
        datapath = self.parameters['raw_data_path']
        bgpath = datapath + "/bg/"
        darkpath = datapath + "/dark/"
        samplepath = datapath + "/sample/"
        # 选定ROI，需要修改的部分
        ROI_X1 = self.parameters['ROI_X1']
        ROI_X2 = self.parameters['ROI_X2']
        ROI_Y1 = self.parameters['ROI_Y1']
        ROI_Y2 = self.parameters['ROI_Y2']

        ROI_height = ROI_Y2 - ROI_Y1;
        ROI_width = ROI_X2 - ROI_X1;
        # ------------------------------------------------------

        # 读取dark,并对所有dark取平均
        sum_pixels = None
        for i in range(Nangles_f):
            for j in range(Ndarkstep):
                for k in range(Ndark):
                    for l in range(Nstages):
                        # 读取图像并矫正
                        darkdir = darkpath + 'bg_0_%d_%d_0_%d.tif' % (j, i, k)
                        # print(darkdir)
                        dark = iio.imread(darkdir)
                        # 初始化sum_pixels，如果是第一张图像
                        if sum_pixels is None:
                            sum_pixels = dark.astype(int)
                        else:
                            # 累加像素值
                            sum_pixels += dark

        # 计算平均暗电流
        average_dark_image = (sum_pixels / (Nangles_f * Ndarkstep * Ndark * Nstages)).astype(np.uint16)

        bgcorrect_roi = np.zeros((Nsteps, ROI_height, ROI_width),dtype = np.uint16)
        for i in range(Nangles_f):
            for j in range(Nsteps):
                # 读取图像并矫正
                bgdir = bgpath + 'bg_0_%d_%d_0_0.tif' % (j, i)
                print(bgdir)
                bg = iio.imread(bgdir)
                bgcorrect = bg - average_dark_image
                bgcorrect_roi[j, :, :] = bgcorrect[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]

        samplecorrect_roi = np.zeros((Nsteps, Nangles, ROI_height, ROI_width), dtype=np.uint16)
        for i in range(Nangles):
            for j in range(Nsteps):
                sampledir = samplepath + 'sample_0_%d_%d_0_0.tif' % (j, i)
                print(sampledir)
                sample = iio.imread(sampledir)
                samplecorrect = sample - average_dark_image
                samplecorrect_roi[j, i, :, :] = samplecorrect[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]

        raw_data = {'flat': bgcorrect_roi, 'tomography': samplecorrect_roi}

        save_data(self.output().path, raw_data)
        ########################################################################################################################

class SignalRetrieval(luigi.Task):
    """
    Do the Fourier Analysis on the raw data
    """
    data_name = luigi.Parameter()
    parameters = load_rec_config(data_name)
    data_in_path  = parameters['processed_data_path']
    data_out_path = parameters['processed_data_path']

    def requires(self):
        return LoadRawData(data_name=self.data_name) 

    def output(self):
        filename = f"{dataset_core_path(self.input().path)}_sino.{FORMAT}"
        return luigi.LocalTarget(filename)

    def run(self):
        logging.debug('SignalRetrieval Task')
           

        raw_data = load_data(self.input().path)
        flat = raw_data['flat']
        phasestep_sinogram = raw_data['tomography']
        (nsteps, nprojections, nheight, nwidth) = phasestep_sinogram.shape

        mask = self.parameters['detector_column_mask'] # need to get from
        for m in mask:
            flat[..., m-2] = flat[..., m-3]
            flat[..., m-1] = flat[..., m-3]
            flat[..., m+1] = flat[..., m+3]
            flat[..., m+2] = flat[..., m+3]
            flat[..., m]  = (flat[..., m-1] + flat[..., m+1]) / 2
            phasestep_sinogram[..., m-2] = phasestep_sinogram[..., m-3]
            phasestep_sinogram[..., m-1] = phasestep_sinogram[..., m-3]
            phasestep_sinogram[..., m+1] = phasestep_sinogram[..., m+3]
            phasestep_sinogram[..., m+2] = phasestep_sinogram[..., m+3]
            phasestep_sinogram[..., m]  = (phasestep_sinogram[..., m-1] + phasestep_sinogram[..., m+1]) / 2

        bin_w = self.parameters['binning_w']
        bin_h = self.parameters['binning_h']

        print('bin_w:',bin_w)
        print('bin_h:', bin_h)


        if (nwidth % bin_w) or (nheight % bin_h):
            logging.error('The binning factor is not valid.')
            raise Exception('The binning factor is not valid.')
        
        if bin_w > 1:
            # binning along the detector width
            flat_tmp = flat[..., 0:nwidth:bin_w]

            for i in range(1, bin_w):
                flat_tmp =  flat_tmp + flat[...,i:nwidth:bin_w]
            flat = flat_tmp
            phasestep_sinogram_tmp = phasestep_sinogram[..., 0:nwidth:bin_w]
            for i in range(1, bin_w):
                phasestep_sinogram_tmp =  phasestep_sinogram_tmp + phasestep_sinogram[...,i:nwidth:bin_w]
            phasestep_sinogram = phasestep_sinogram_tmp
        # binning along the detector height


        if bin_h > 1:
            flat_tmp = flat[:, 0:nheight:bin_h, :]

            for i in range(1, bin_h):
                flat_tmp =  flat_tmp + flat[:,i:nheight:bin_h, :]

            flat = flat_tmp

            phasestep_sinogram_tmp = phasestep_sinogram[..., 0:nheight:bin_h, :]
            for i in range(1, bin_w):
                print(phasestep_sinogram_tmp.shape)
                print(phasestep_sinogram[..., i:nheight:bin_h, :].shape)
                phasestep_sinogram_tmp =  phasestep_sinogram_tmp + phasestep_sinogram[..., i:nheight:bin_h, :]
            phasestep_sinogram = phasestep_sinogram_tmp

        flat_fft = np.fft.rfft(flat, axis=0)
        flat_abs = abs(flat_fft)[0,...]
        flat_phase = np.angle(flat_fft)[1,...]
        flat_vis = 2 * abs(flat_fft)[1,...] / flat_abs

        phasestep_sinogram_fft = np.fft.rfft(phasestep_sinogram, axis=0) 
        del phasestep_sinogram
        gc.collect()


        rsinogram_abs   = abs(phasestep_sinogram_fft[0,...])
        rsinogram_phase = np.angle(phasestep_sinogram_fft[1,...])
        rsinogram_vis   = 2 * abs(phasestep_sinogram_fft[1,...]) / rsinogram_abs

        s_abs = -np.log(rsinogram_abs / flat_abs)

        s_dpc = np.angle(np.exp(1j*(rsinogram_phase - flat_phase)))
        s_dci = -np.log(rsinogram_vis / flat_vis)

        del rsinogram_abs
        del rsinogram_phase
        del rsinogram_vis
        gc.collect()

        s_abs[np.isnan(s_abs)] = 0
        s_dci[np.isnan(s_dci)] = 0
        s_abs[np.isinf(s_abs)] = 0
        s_dci[np.isinf(s_dci)] = 0

        #相位解卷绕
        pi = math.pi
        s_dpc= wrapphse(s_dpc)
        print('开始矫正背景相衬')
        s_dpc= np.swapaxes(s_dpc, 0, 2)
        s_dpc= bgcorrect2(s_dpc)
        s_dpc= np.swapaxes(s_dpc, 0, 2)
        print('开始矫正背景吸收')
        s_abs= np.swapaxes(s_abs, 0, 2)
        s_abs= bgcorrect2(s_abs)
        s_abs= np.swapaxes(s_abs, 0, 2)
        print('开始矫正背景暗场')
        s_dci= np.swapaxes(s_dci, 0, 2)
        s_dci= np.swapaxes(s_dci, 0, 2)

        sino = {'absorption' : s_abs, 'dpc': s_dpc, 'dark_field': s_dci, 'visibility_map': flat_vis, 'phase_map': flat_phase, 'intensity_map': flat_abs}
        save_data(self.output().path, sino)

class SinoProcessing(luigi.Task, TimeTaskMixin):
    """
    Pre processing the sinograms: finding the rotation center; cropping; masking
    """
    data_name = luigi.Parameter()
    parameters = load_rec_config(data_name)
    data_in_path  = parameters['processed_data_path']
    data_out_path = parameters['processed_data_path'] 

    def requires(self):
        return SignalRetrieval(data_name=self.data_name) 

    def output(self):
        filename = f"{dataset_core_path(self.input().path)}_sino_preprocessed.{FORMAT}"
        return luigi.LocalTarget(filename)

    def run(self):
        logger = logging.getLogger()
        logger.debug('====== SinoPreProcessing Task ======')
        sino = load_data(self.input().path)

        #step 1: applying detector global mask
        # this is done in the SignalRetrieval task, applying to the detector raw data

        #针对老平台数据，需要做旋转
        sino['absorption'] = np.rot90(sino['absorption'], 3, (1, 2))
        sino['dpc'] = np.rot90(sino['dpc'], 3, (1, 2))
        sino['dark_field'] = np.rot90(sino['dark_field'], 3, (1, 2))

        #step 2: choose the right projection for rec
        projtrim = self.parameters['proj_trim']

        #step 3: cropping first
        htrim = self.parameters['detector_trim']

        # do the trimming
        sino['absorption'] = sino['absorption'][projtrim,:,htrim]
        sino['dpc']        = sino['dpc'][projtrim,:,htrim]
        sino['dark_field'] = sino['dark_field'][projtrim,:,htrim]

        #step 4: finding the rotation center
        center_of_rotation = self.parameters['rotation_center']
        (nangle, nslice, nwidth) = sino['absorption'].shape

        #我自己写的,其实就相当于寻找整张图片的旋转轴
        id_slice = self.parameters['slice_index']
        a = int(nangle/2)
        b=  int(nangle/4)
        image0_0 = sino['absorption'][0,:,:];
        image180_0 = sino['absorption'][a,:,:];
        image0_1 = sino['absorption'][b,:,:];
        image180_1 = sino['absorption'][b+a,:,:];
        center_of_rotation_0 = int(find_center_pc(image0_0,image180_0))
        center_of_rotation_1 = int(find_center_pc(image0_1,image180_1))
        center_of_rotation = int((center_of_rotation_0+center_of_rotation_1)/2)


        if nwidth - center_of_rotation > center_of_rotation:
            w = center_of_rotation
        else:
            w = nwidth - center_of_rotation

        print('rotate_center:')
        print(center_of_rotation)
        # cropping the sinogram
        idx = np.s_[(center_of_rotation-w):(center_of_rotation+w)]
        sino['absorption'] = sino['absorption'][...,idx]
        sino['dpc']        = sino['dpc'][...,idx]
        sino['dark_field'] = sino['dark_field'][...,idx]

        #step 5: denoising, 太费时间了，算了
        # sino['absorption'] = scipy.signal.medfilt(sino['absorption'],kernel_size=(3, 1, 3))
        # sino['dpc']        = scipy.signal.medfilt(sino['dpc'],kernel_size=(3, 1, 3))
        # sino['dark_field'] = scipy.signal.medfilt(sino['dark_field'], kernel_size=(3, 1, 3))

        #step 6：去环状伪影，费时间，但是效果不错，你们可以跑跑
        # 去环状伪影
        # print('开始去吸收图环状伪影')
        # sino['absorption'] = tomopy.prep.stripe.remove_stripe_fw(sino['absorption'], level=None, wname='db5', sigma=2, pad=True, ncore=None,nchunk=None)  # 用的这个小波变换去伪影
        # print('去吸收图环状伪影成功')
        # print('开始去相衬图环状伪影')
        # sino['dpc'] = tomopy.prep.stripe.remove_stripe_fw(sino['dpc'], level=None, wname='db5', sigma=2, pad=True, ncore=None, nchunk=None)  # 用的这个小波变换去伪影
        # print('去相衬图环状伪影成功')
        # print('开始去暗场图环状伪影')
        # sino['dark_field'] = tomopy.prep.stripe.remove_stripe_fw(sino['dark_field'], level=None, wname='db5', sigma=2, pad=True, ncore=None, nchunk=None)  # 用的这个小波变换去伪影
        # print('去暗场图环状伪影成功')

        #save the data again
        sino = {'absorption': sino['absorption'],'dpc': sino['dpc'], 'dark_field': sino['dark_field'],  'center_of_rotation': center_of_rotation}
        save_data(self.output().path, sino)


class Reconstruction(luigi.Task, TimeTaskMixin):
    """
    Do the 3D reconstruction of the data
    """
    data_name = luigi.Parameter()
    parameters = load_rec_config(data_name)
    data_in_path  = parameters['processed_data_path']
    data_out_path = parameters['processed_data_path']
    targets = parameters['targets']
    pixel_size = parameters["pixel_size"]
    


    def requires(self):
        return SinoProcessing(data_name=self.data_name) 
   
    def output(self):
        filename = f"{dataset_core_path(self.input().path)}_rec.{FORMAT}"
        return luigi.LocalTarget(filename)
    
    def run(self):
        '''
            TODO 
                - the reconstruction tasks can be further parallelised, just need to make sure the GPU has enough momoeny to handle it
                - the recon code can be organized in a better way
        '''
        logging.debug('Reconstruction Task')

        sino = load_data(self.input().path)

        sino_abs = sino['absorption']
        # ================== 数据预处理 ==================
        (nangle, nslice, nwidth) = sino_abs.shape #(720, 484, 402)    #binning=2:(720, 242, 296)
        print(f"sino_abs.shape:{sino_abs.shape}")
        p = np.ascontiguousarray(np.transpose(sino_abs, (1,2,0))) # (484, 402, 720) binning=2:(242, 296, 720)
        print(f"p.shape:{p.shape}")
        print(p[100,100,100])
        pixel_size = self.parameters["pixel_size"]
        angle_step = self.parameters['angle_step']
        SOD = self.parameters['SOD']
        ODD = self.parameters['ODD']
        magnification_factor = (SOD + ODD)/SOD
        Vitualdetectorinterval = pixel_size
        Pictureinterval = pixel_size/ magnification_factor
        angles = np.arange(nangle) * angle_step
        # # 转换参数
        # nwidth = np.unit32(nwidth)
        # nslice = np.unit32(nslice)
        # angles = angles.astype(np.float32)
        # Vitualdetectorinterval = np.float32(Vitualdetectorinterval)
        # SOD = np.float32(SOD)
        # ODD = np.float32(ODD)
        # R = SOD + ODD
        # pixel_size = np.float32(pixel_size)
     
        
        # 在调用核函数前添加类型检查
        print("参数类型验证:")
        print("nslice type:", type(nslice))          # 应显示 np.int32
        print("SOD type:", type(SOD))                # 应显示 np.float32
        print("angles dtype:", angles.dtype)         # 应显示 float32
                
        # Compile CUDA kernels
        module_weight = SourceModule(weight_kernel_code)
        module_backprojection = SourceModule(backprojection_kernel_code)
        
        # Get kernel functions
        weight_kernel = module_weight.get_function("apply_cone_beam_weighting")
        backproj_kernel = module_backprojection.get_function("cone_beam_backprojection")
        
        # Convert projection data to float32 for GPU
        p = p.astype(np.float32)
        
        # Allocate GPU memory for projection data
        p_gpu = cuda.to_device(p.flatten())
        p_weighted_gpu = cuda.mem_alloc(p.nbytes)
        
        
        # Step 1: Apply cone-beam weighting factor using CUDA
        print("Starting cone-beam weighting on GPU...")
        
        # Define grid and block dimensions for weighting kernel
        block_dim = (16, 16, 1)  # 16x16 threads per block
        grid_dim = (
            math.ceil(nslice / block_dim[0]),
            math.ceil(nwidth / block_dim[1]),
            nangle
        )
        
        # Launch weighting kernel
        weight_kernel(
            p_gpu, p_weighted_gpu,
            np.int32(nslice), np.int32(nwidth),
            np.float32(SOD), np.float32(Vitualdetectorinterval),
            block=block_dim,
            grid=grid_dim
        )
        
        # Copy weighted projections back to host for filtering
        p_weighted = np.empty_like(p)
        cuda.memcpy_dtoh(p_weighted, p_weighted_gpu)
        p_weighted = p_weighted.reshape(p.shape)
        
        print("Cone-beam weighting completed on GPU")
        
        # Step 2: Convolve with Ram-Lak filter row by row on CPU
        # (convolution can be challenging to implement efficiently on GPU for this case)
        print("Starting convolution on CPU...")
        RL_list = RL_fun(nwidth, Vitualdetectorinterval)
        p_conv1 = np.zeros_like(p_weighted, dtype=np.float32)
        
        for k in range(nangle):
            for i in range(nslice):
                p_conv1[i, :, k] = scipy.signal.convolve(p_weighted[i, :, k], RL_list, mode='same')
        
        print("Convolution completed on CPU")
        
        # Flatten and transfer filtered projections to GPU
        p_conv1_gpu = cuda.to_device(p_conv1.flatten())
        
        # Allocate memory for reconstruction volume on GPU
        volume_size = nwidth * nwidth * nslice * np.dtype(np.float32).itemsize
        volume_gpu = cuda.mem_alloc(volume_size)
        cuda.memset_d32(volume_gpu, 0, nwidth * nwidth * nslice)
        
        # Transfer angles to GPU
        angles_gpu = cuda.to_device(angles)
        
        # Step 3: Backprojection on GPU
        print("Starting backprojection on GPU...")
        
        # Define block and grid dimensions for backprojection
        block_dim_bp = (8, 8, 1)  # 8x8 threads per block
        grid_dim_bp = (
            math.ceil(nwidth / block_dim_bp[0]),
            math.ceil(nwidth / block_dim_bp[1]),
            nslice
        )
        
        # Launch backprojection kernel
        backproj_kernel(
            p_conv1_gpu, volume_gpu,
            np.int32(nwidth), np.int32(nslice), np.int32(nangle),
            angles_gpu, np.float32(SOD), np.float32(Vitualdetectorinterval),
            np.float32(Pictureinterval), np.float32(angle_step * np.pi / 180),
            block=block_dim_bp,
            grid=grid_dim_bp
        )
        
        # Copy result back to host
        RePic = np.zeros((nwidth, nwidth, nslice), dtype=np.float32)
        cuda.memcpy_dtoh(RePic.flatten(), volume_gpu)
        
        print("Backprojection completed on GPU")
        
        # # Apply rotation transforms to match the expected orientation (on CPU)
        # print("Applying final transformations...")
        # Reconstruction = np.zeros_like(RePic)
        
        # # First rotation
        # Shepplogan1 = np.zeros_like(RePic)
        # for i in range(nwidth):
        #     Shepplogan1[:, i, :] = np.rot90(np.squeeze(RePic[:, i, :]).T, 2)
        
        # # Second rotation
        # Shepplogan2 = np.zeros_like(Shepplogan1)
        # for i in range(nwidth):
        #     Shepplogan2[i, :, :] = np.flipud(np.squeeze(Shepplogan1[i, :, :]))
        
        # # Final orientation
        # for i in range(nwidth):
        #     Reconstruction[:, i, :] = np.flipud(np.squeeze(Shepplogan2[:, i, :]))
        
  
        RePic = np.ascontiguousarray(np.transpose(RePic, (2,0,1))) 
        abs_reconstruction =  RePic

        dci_reconstruction = []

        dpc_reconstruction = []

        rec = {'absorption': abs_reconstruction, 'phase_contrast': dpc_reconstruction, 'dark_field': dci_reconstruction}
        save_data(self.output().path, rec) 
        logging.debug('FDK reconstruction is done')
    

class ReconSingleSlice(luigi.Task, TimeTaskMixin):
    """
    Do the 3D reconstruction of the data
    """
    data_name = luigi.Parameter()
    parameters = load_rec_config(data_name)
    data_in_path  = parameters['processed_data_path']
    data_out_path = parameters['processed_data_path']
    targets = parameters['targets']
    pixel_size = parameters["pixel_size"]


    def requires(self):
        return SinoProcessing(data_name=self.data_name) 
   
    def output(self):
        filename = f"{dataset_core_path(self.input().path)}_singleslice.{FORMAT}"
        return luigi.LocalTarget(filename)
    
    def run(self):
        '''
            TODO 
                - the reconstruction tasks can be further parallelised, just need to make sure the GPU has enough momoeny to handle it
                - the recon code can be organized in a better way
        '''
        logging.debug('Reconstruction Task')

        sino = load_data(self.input().path)

        sino_abs = sino['absorption']
        print(f"sino_abs.shape1:{sino_abs.shape}") #(720, 242, 296)
        sino_abs = sino_abs.swapaxes( 0 , 1 )  # (nslice, nangle,  nwidth)
        (nangle, nslice, nwidth) = sino_abs.shape 
        sino_slice = np.zeros((1, nslice, nwidth ),dtype = np.float32)
        id_slice = self.parameters['slice_index']
        sino_slice[0,:,:] = sino_abs[id_slice,:,:]
        
        # ================== 数据预处理 ==================
        (nslice, nangle,  nwidth) = sino_slice.shape 
        print(f"sino_abs.shape2:{sino_abs.shape}") #(720, 242, 296)
        p = np.ascontiguousarray(np.transpose(sino_abs, (0,2,1))) # ( nslice, nwidth, nangle)
        print(f"p.shape:{p.shape}")
        print(p[100,100,100])
        pixel_size = self.parameters["pixel_size"]
        angle_step = self.parameters['angle_step']
        SOD = self.parameters['SOD']
        ODD = self.parameters['ODD']
        magnification_factor = (SOD + ODD)/SOD
        Vitualdetectorinterval = pixel_size
        Pictureinterval = pixel_size/ magnification_factor
        angles = np.arange(nangle) * angle_step

        
        # 在调用核函数前添加类型检查
        print("参数类型验证:")
        print("nslice type:", type(nslice))          # 应显示 np.int32
        print("SOD type:", type(SOD))                # 应显示 np.float32
        print("angles dtype:", angles.dtype)         # 应显示 float32
                
        # Compile CUDA kernels
        module_weight = SourceModule(weight_kernel_code)
        module_backprojection = SourceModule(backprojection_kernel_code)
        
        # Get kernel functions
        weight_kernel = module_weight.get_function("apply_cone_beam_weighting")
        backproj_kernel = module_backprojection.get_function("cone_beam_backprojection")
        
        # Convert projection data to float32 for GPU
        p = p.astype(np.float32)
        
        # Allocate GPU memory for projection data
        p_gpu = cuda.to_device(p.flatten())
        p_weighted_gpu = cuda.mem_alloc(p.nbytes)
        
        
        # Step 1: Apply cone-beam weighting factor using CUDA
        print("Starting cone-beam weighting on GPU...")
        
        # Define grid and block dimensions for weighting kernel
        block_dim = (16, 16, 1)  # 16x16 threads per block
        grid_dim = (
            math.ceil(nslice / block_dim[0]),
            math.ceil(nwidth / block_dim[1]),
            nangle
        )
        
        # Launch weighting kernel
        weight_kernel(
            p_gpu, p_weighted_gpu,
            np.int32(nslice), np.int32(nwidth),
            np.float32(SOD), np.float32(Vitualdetectorinterval),
            block=block_dim,
            grid=grid_dim
        )
        
        # Copy weighted projections back to host for filtering
        p_weighted = np.empty_like(p)
        cuda.memcpy_dtoh(p_weighted, p_weighted_gpu)
        p_weighted = p_weighted.reshape(p.shape)
        
        print("Cone-beam weighting completed on GPU")
        
        # Step 2: Convolve with Ram-Lak filter row by row on CPU
        # (convolution can be challenging to implement efficiently on GPU for this case)
        print("Starting convolution on CPU...")
        RL_list = RL_fun(nwidth, Vitualdetectorinterval)
        p_conv1 = np.zeros_like(p_weighted, dtype=np.float32)
        
        for k in range(nangle):
            for i in range(nslice):
                p_conv1[i, :, k] = scipy.signal.convolve(p_weighted[i, :, k], RL_list, mode='same')
        
        print("Convolution completed on CPU")
        
        # Flatten and transfer filtered projections to GPU
        p_conv1_gpu = cuda.to_device(p_conv1.flatten())
        
        # Allocate memory for reconstruction volume on GPU
        volume_size = nwidth * nwidth * nslice * np.dtype(np.float32).itemsize
        volume_gpu = cuda.mem_alloc(volume_size)
        cuda.memset_d32(volume_gpu, 0, nwidth * nwidth * nslice)
        
        # Transfer angles to GPU
        angles_gpu = cuda.to_device(angles)
        
        # Step 3: Backprojection on GPU
        print("Starting backprojection on GPU...")
        
        # Define block and grid dimensions for backprojection
        block_dim_bp = (8, 8, 1)  # 8x8 threads per block
        grid_dim_bp = (
            math.ceil(nwidth / block_dim_bp[0]),
            math.ceil(nwidth / block_dim_bp[1]),
            nslice
        )
        
        # Launch backprojection kernel
        backproj_kernel(
            p_conv1_gpu, volume_gpu,
            np.int32(nwidth), np.int32(nslice), np.int32(nangle),
            angles_gpu, np.float32(SOD), np.float32(Vitualdetectorinterval),
            np.float32(Pictureinterval), np.float32(angle_step),
            block=block_dim_bp,
            grid=grid_dim_bp
        )
        
        # Copy result back to host
        RePic = np.zeros((nwidth, nwidth, nslice), dtype=np.float32)
        cuda.memcpy_dtoh(RePic.flatten(), volume_gpu)
        
        print("Backprojection completed on GPU")
        
  
        RePic = np.ascontiguousarray(np.transpose(RePic, (2,0,1))) 
        abs_reconstruction =  RePic

        dci_reconstruction = []

        dpc_reconstruction = []

        rec = {'absorption': abs_reconstruction, 'phase_contrast': dpc_reconstruction, 'dark_field': dci_reconstruction}
        save_data(self.output().path, rec) 
        logging.debug('FDK reconstruction is done')
       
        

class AbstractCopy(luigi.Task, TimeTaskMixin):
    data_name = luigi.Parameter()
    parameters = load_rec_config(data_name)
    resources = { "network" : 1 }
   
    def output(self):
        filename = os.path.join(
            self.parameters['destination_data_path'],
            os.path.split(self.input().path)[-1])
        return luigi.LocalTarget(filename)

    def run(self):
        shutil.copy(self.input().path, self.output().path)


class LoadRawDataCopy(AbstractCopy):
    def requires(self):
        return LoadRawData(data_name=self.data_name)


class SignalRetrievalCopy(AbstractCopy):
    def requires(self):
        return SignalRetrieval(data_name=self.data_name)


class SinoProcessingCopy(AbstractCopy):
    def requires(self):
        return SinoProcessing(data_name=self.data_name)


class ReconstructionCopy(AbstractCopy):
    def requires(self):
        return Reconstruction(data_name=self.data_name)


class AllCopy(luigi.Task, TimeTaskMixin):
    data_name = luigi.Parameter()

    def requires(self):
        return {
            "LoadRawDataCopy": LoadRawDataCopy(data_name=self.data_name),
            "SignalRetrievalCopy": SignalRetrievalCopy(data_name=self.data_name),
            "SinoProcessingCopy": SinoProcessingCopy(data_name=self.data_name),
            "ReconstructionCopy": ReconstructionCopy(data_name=self.data_name),
        }


class LoadNonPSRawData(luigi.Task):
    """
    Dirty task to handle special scans times. 5 scans without vertical movements. The meta files are different. 
    """
    data_name   = luigi.Parameter()
    parameters  = load_rec_config(data_name)
    data_in_path  = parameters['raw_data_path']
    data_out_path = parameters['processed_data_path']
    resources = { "network" : 1 }
    
    def output(self):
        """
        Write the data into an easy to handle file
        """
        filename = os.path.join(self.data_out_path, f"{self.data_name}_raw.{FORMAT}")
        return luigi.LocalTarget(filename)
    
    def run(self):
        """
        Load and sort out the data
        """
        logging.debug('LoadNoPSRawData Task')

        ia.STORAGE = self.data_in_path
        raw_data = load_non_ps_ffc_tomography(self.data_name)

        # shuffle the raw data
        #flat = raw_data['flats']
        flat = np.mean(raw_data["flats"], axis=1)
        flat = flat[1,...]
        flat = flat[None,...]
        flat = np.repeat(flat, 5, axis=0)
        # flat = np.array([ 
        #     np.mean([
        #         np.mean([ a for a in flat_series[50:] ], axis=0)
        #         for flat_series in ffc_tomo_series["flats"] ], axis=0)
        #     for ffc_tomo_series in raw_data ])
        # the sample data will be a 4D dataset: (height, width, steps, projections)

        #TODO
        #scan correction, this is different from each scans probably
        #processing each projections
        # number of lines
        start_shifts = np.r_[0, 0, 0, 0, 0]
        start_shifts -= np.min(start_shifts)
        nprojections = len(raw_data["tomography"][0])
        # the flat data should be a 3D dataset: (steps, projection height, width)
        phasestep_sinograms = raw_data['tomography']
        # phasestep_sinograms = np.array([
        #     [ tomo_scan[
        #             start_shift :
        #             (nprojections - np.max(start_shifts) + start_shift)]
        #         for tomo_scan
        #         in ffc_tomo_series["tomography"] ]
        #     for ffc_tomo_series, start_shift
        #     in zip(raw_data, start_shifts) ])


        raw_data = {'flat': flat, 'tomography': phasestep_sinograms}
        save_data(self.output().path, raw_data)
    
    
    
# CUDA kernel for cone-beam weighting
weight_kernel_code = """
__global__ void apply_cone_beam_weighting(float* projection, float* weighted, 
                                         int nslice, int nwidth,
                                         float SOD, float detector_interval) 
{
    // Calculate global indices
    const int v_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int u_idx = blockIdx.y * blockDim.y + threadIdx.y;
    const int angle_idx = blockIdx.z;
    
    if (v_idx < nslice && u_idx < nwidth) {
        // Calculate detector positions relative to center
        const float u = (u_idx - (nwidth/2.0f - 0.5f)) * detector_interval;
        const float v = (v_idx - (nslice/2.0f - 0.5f)) * detector_interval;
        
        // Calculate weighting factor
        const float weight = SOD / sqrtf(SOD*SOD + u*u + v*v);
        
        // Apply weight to projection
        const int idx = v_idx * nwidth + u_idx + angle_idx * nslice * nwidth;
        weighted[idx] = projection[idx] * weight;
    }
}
"""

# CUDA kernel for backprojection
backprojection_kernel_code = """
__global__ void cone_beam_backprojection(float* p_conv, float* volume,
                                        int nwidth, int nslice, int nangle,
                                        float* angles, float SOD, float detector_interval,
                                        float voxel_size, float angle_step_rad)
{
    // Calculate global indices for volume
    const int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int y_idx = blockIdx.y * blockDim.y + threadIdx.y;
    const int z_idx = blockIdx.z;
    
    if (x_idx >= nwidth || y_idx >= nwidth || z_idx >= nslice)
        return;
        
    // Calculate physical coordinates
    const float x = (-x_idx + nwidth/2.0f + 0.5f) * voxel_size;
    const float y = (-y_idx + nwidth/2.0f + 0.5f) * voxel_size;
    const float z = (-z_idx + nslice/2.0f + 0.5f) * voxel_size;
    
    float fxyz = 0.0f;
    
    // Loop over all projection angles
    for (int beta_idx = 0; beta_idx < nangle; beta_idx++) {
        const float beta = angles[beta_idx];
        
        // Calculate source position and ray direction
        const float denom = SOD + x * cosf(beta) + y * sinf(beta);
        
        // Skip if point is behind or too close to the source
        if (denom <= 0.1f)
            continue;
            
        // Calculate the projection of the point onto the detector
        const float U_source = SOD * (-x * sinf(beta) + y * cosf(beta)) / denom;
        const float V_source = z * SOD / denom;
        
        // Calculate detector indices
        const int u_idx = int(nwidth/2.0f - U_source/detector_interval);
        const int v_idx = int(nslice/2.0f - V_source/detector_interval);
        
        // Check if projection is within detector bounds
        if (u_idx >= 0 && u_idx < nwidth && v_idx >= 0 && v_idx < nslice) {
            // Apply cone-beam magnification factor
            const float magnification = (SOD / denom) * (SOD / denom);
            
            // Get filtered projection value
            const int proj_idx = v_idx * nwidth + u_idx + beta_idx * nslice * nwidth;
            
            // Add contribution from this angle
            fxyz += magnification * p_conv[proj_idx] * angle_step_rad;
        }
    }
    
    // Store the reconstructed value
    const int vol_idx = x_idx + y_idx * nwidth + z_idx * nwidth * nwidth;
    volume[vol_idx] = fxyz * detector_interval ;
}
"""


def RL_fun(DetNum, VitualDetInt):
    """Create Ram-Lak filter in spatial domain"""
    RL_list = np.zeros(2 * DetNum - 1)
    for i in range(2 * DetNum - 1):
        if (i - DetNum) % 2 == 0:
            RL_list[i] = 0
        else:
            RL_list[i] = -1 / (2 * np.pi**2 * VitualDetInt**2 * (i - DetNum)**2)
    
    RL_list[DetNum - 1] = 1 / (8 * VitualDetInt**2)
    return RL_list
