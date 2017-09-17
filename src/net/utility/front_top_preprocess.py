#!/usr/bin/env python
# -*- coding:UTF-8 -*-

from __future__ import print_function 
from __future__ import absolute_import 
from __future__ import division
import pycuda.driver as cuda 
from pycuda.compiler import SourceModule 
import numpy as np
from config import *
import os

# for old version titan x, the same as 1080
# pycuda._driver.device_attribute.MAX_THREADS_PER_BLOCK: 1024
# pycuda._driver.device_attribute.MAX_BLOCK_DIM_X: 1024
# pycuda._driver.device_attribute.MAX_BLOCK_DIM_Y: 1024
# pycuda._driver.device_attribute.MAX_BLOCK_DIM_Z: 64
# pycuda._driver.device_attribute.MAX_GRID_DIM_X: 2147483647
# pycuda._driver.device_attribute.MAX_GRID_DIM_Y: 65535
# pycuda._driver.device_attribute.MAX_GRID_DIM_Z: 65535

module_buff = b"".join(open(os.path.join(cfg.ROOT_DIR, 'src/net/utility', 'front_top_kernel.cubin'), 'rb').readlines())

def get_gpu_info():
    cuda.init()
    device = cuda.Device(0)
    print(device.get_attributes())

def lidar_to_top_cuda(lidar):
    # input:
    # lidar: (N, 4) 4->(x,y,z,i) in lidar coordinate
    lidar = np.copy(lidar)
    mod = cuda.module_from_buffer(module_buff)
    func = mod.get_function('_Z12lidar_to_topPfPiS0_S0_S_S_S0_')
    func_density = mod.get_function('_Z20lidar_to_top_densityPfPiS0_S0_S0_')
    # trunc 
    idx = np.where (lidar[:,0]>TOP_X_MIN)
    lidar = lidar[idx]
    idx = np.where (lidar[:,0]<TOP_X_MAX)
    lidar = lidar[idx]

    idx = np.where (lidar[:,1]>TOP_Y_MIN)
    lidar = lidar[idx]
    idx = np.where (lidar[:,1]<TOP_Y_MAX)
    lidar = lidar[idx]

    idx = np.where (lidar[:,2]>TOP_Z_MIN)
    lidar = lidar[idx]
    idx = np.where (lidar[:,2]<TOP_Z_MAX)
    lidar = lidar[idx]
    # shape 
    X0, Xn = 0, int((TOP_X_MAX-TOP_X_MIN)//TOP_X_DIVISION)+1
    Y0, Yn = 0, int((TOP_Y_MAX-TOP_Y_MIN)//TOP_Y_DIVISION)+1
    Z0, Zn = 0, int((TOP_Z_MAX-TOP_Z_MIN)/TOP_Z_DIVISION)
    height  = Xn - X0
    width   = Yn - Y0
    channel = Zn - Z0  + 2
    # intensity and density channel do not cal seperately in kernel function 
    top = np.zeros(shape=(height,width,channel), dtype=np.float32)
    top_density = np.zeros(shape=(height,width,1), dtype=np.float32)
    top_shape = np.array(top.shape).astype(np.int32)
    lidar_shape = np.array(lidar.shape).astype(np.int32)

    # voxelize lidar 
    lidar[:, 0] = ((lidar[:, 0]-TOP_X_MIN)//TOP_X_DIVISION).astype(np.int32)
    lidar[:, 1] = ((lidar[:, 1]-TOP_Y_MIN)//TOP_Y_DIVISION).astype(np.int32)
    lidar[:, 2] = (lidar[:, 2]-TOP_Z_MIN)/TOP_Z_DIVISION

    lidar = lidar[np.lexsort((lidar[:, 2], lidar[:, 1], lidar[:, 0])), :]
    lidar_x = np.ascontiguousarray(lidar[:, 0].astype(np.int32))
    lidar_y = np.ascontiguousarray(lidar[:, 1].astype(np.int32))
    lidar_z = np.ascontiguousarray(lidar[:, 2])
    lidar_i = np.ascontiguousarray(lidar[:, 3])
    
    func(
        cuda.InOut(top), 
        cuda.In(top_shape),
        cuda.In(lidar_x), 
        cuda.In(lidar_y), 
        cuda.In(lidar_z), 
        cuda.In(lidar_i), 
        cuda.In(lidar_shape),
        #intensity and density channel do not cal seperately
        block=(channel, 1, 1),  # a thread <-> a channel 
        grid=(int(lidar_shape[0]), 1, 1)  # a grid <-> a point in laser scan  
    )
    func_density(
        cuda.InOut(top_density),
        cuda.In(lidar_x),
        cuda.In(lidar_y),
        cuda.In(lidar_shape),
        cuda.In(top_shape),
        block=(1,1,1),
        grid=(1,1,1)
    )
    top_density = (np.log(top_density.astype(np.int32) + 1)/math.log(32)).clip(max=1).astype(np.float32)
    return np.dstack([top[:, :, :-1], top_density])

def lidar_to_front_cuda(lidar):
    # input:
    # lidar: (N, 4) 4->(x,y,z,i) in lidar coordinate

    mod = cuda.module_from_buffer(module_buff)
    func = mod.get_function('_Z14lidar_to_frontPfPiS_S0_')
    # shape 
    channel = 3
    front = np.zeros(shape=(cfg.FRONT_WIDTH, cfg.FRONT_WIDTH, channel), dtype=np.float32)
    front_shape = np.array(front.shape).astype(np.int32)
    lidar_shape = np.array(lidar.shape).astype(np.int32)

    func(
        cuda.InOut(front), 
        cuda.In(front_shape),
        cuda.In(lidar), 
        cuda.In(lidar_shape),
        block=(channel, 1, 1),  # a thread <-> a channel
        grid=(int(lidar_shape[0]), 1, 1)  # a grid <-> a point in laser scan 
    )
    return front

if __name__ == '__main__':
    import time
    import random
    import numpy as np
    from data import lidar_to_top, Preprocess 
    pro = Preprocess()
    
    import pycuda.autoinit

    lidar = np.fromfile('/data/mxj/kitti/object/training/velodyne/007480.bin', dtype=np.float32)
    lidar = lidar.reshape((-1, 4))
    t0 = time.time()
    top = lidar_to_top_cuda(lidar) 
    t1 = time.time()  
    print('done top, {}'.format(t1-t0))
    top_gt = lidar_to_top(lidar)
    t2 = time.time() 
    print('done top cpu, {}'.format(t2-t1))
    front = lidar_to_front_cuda(lidar) 
    t3 = time.time()
    print('done front, {}'.format(t3-t2))
    front_gt = pro.lidar_to_front(lidar)
    t4 = time.time() 
    print('done front cpu, {}'.format(t4-t3))
    from IPython import embed; embed()
