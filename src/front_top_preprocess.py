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
    func_add_points = mod.get_function('_Z25lidar_to_front_add_pointsPiS_S_S_')
    func_fill_front = mod.get_function('_Z25lidar_to_front_fill_frontPfS_PiS0_')
    def cal_height(points):
        return np.clip(points[:, 2] + cfg.VELODYNE_HEIGHT, a_min=0, a_max=None).astype(np.float32).reshape((-1, 1))
    def cal_distance(points):
        return np.sqrt(np.sum(points**2, axis=1)).astype(np.float32).reshape((-1, 1))
    def cal_intensity(points):
        return points[:, 3].astype(np.float32).reshape((-1, 1))
    def to_front(points):
        return np.array([
            np.arctan2(points[:, 1], points[:, 0])/cfg.VELODYNE_ANGULAR_RESOLUTION,
            np.arctan2(points[:, 2], np.sqrt(points[:, 0]**2 + points[:, 1]**2)) \
                /cfg.VELODYNE_VERTICAL_RESOLUTION
        ], dtype=np.int32).T

    # using the same crop method as top view
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

    points = to_front(lidar)
    ind = np.where(cfg.FRONT_C_MIN < points[:, 0])
    points, lidar = points[ind], lidar[ind]
    ind = np.where(points[:, 0] < cfg.FRONT_C_MAX)
    points, lidar = points[ind], lidar[ind]
    ind = np.where(cfg.FRONT_R_MIN < points[:, 1])
    points, lidar = points[ind], lidar[ind]
    ind = np.where(points[:, 1] < cfg.FRONT_R_MAX)
    points, lidar = points[ind], lidar[ind]

    points[:, 0] += int(cfg.FRONT_C_OFFSET)
    points[:, 1] += int(cfg.FRONT_R_OFFSET)
    #points //= 2

    ind = np.where(0 <= points[:, 0])
    points, lidar = points[ind], lidar[ind]
    ind = np.where(points[:, 0] < cfg.FRONT_WIDTH)
    points, lidar = points[ind], lidar[ind]
    ind = np.where(0 <= points[:, 1])
    points, lidar = points[ind], lidar[ind]
    ind = np.where(points[:, 1] < cfg.FRONT_HEIGHT)
    points, lidar = points[ind], lidar[ind]

    # sort for mem friendly 
    idx = np.lexsort((points[:, 1], points[:, 0]))
    points = points[idx, :]
    lidar = lidar[idx, :]

    channel = 3 # height, distance, intencity
    front = np.zeros((cfg.FRONT_WIDTH, cfg.FRONT_HEIGHT, channel), dtype=np.float32)
    weight_mask = np.zeros_like(front[:, :, 0]).astype(np.int32)
    # def _add(x):
    #     weight_mask[int(x[0]), int(x[1])] += 1
    # def _fill(x):
    #     front[int(x[0]), int(x[1]), :] += x[2:]
    # np.apply_along_axis(_add, 1, points)
    buf = np.hstack((points, cal_height(lidar), cal_distance(lidar), cal_intensity(lidar))).astype(np.float32)
    # np.apply_along_axis(_fill, 1, buf)
    
    func_add_points(
        cuda.InOut(weight_mask),  
        cuda.In(points),
        cuda.In(np.array(weight_mask.shape).astype(np.int32)),
        cuda.In(np.array(points.shape).astype(np.int32)),
        block=(1, 1, 1), 
        grid=(1, 1, 1), # points
    )
    weight_mask[weight_mask == 0] = 1  # 0 and 1 are both 1
    func_fill_front(
        cuda.InOut(front),
        cuda.In(buf),
        cuda.In(np.array(front.shape).astype(np.int32)),
        cuda.In(np.array(buf.shape).astype(np.int32)),
        block=(3, 1, 1), # channel 
        grid=(1, 1, 1)  # points 
    )

    front /= weight_mask[:, :, np.newaxis]
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
    front_gt = pro.lidar_to_front_fast(lidar)
    t4 = time.time() 
    print('done front cpu, {}'.format(t4-t3))
