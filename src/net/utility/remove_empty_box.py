from __future__ import print_function 
from __future__ import absolute_import 
from __future__ import division
import pycuda.driver as cuda 
import numpy as np
from config import cfg
import os

# for old version titan x, the same as 1080
# pycuda._driver.device_attribute.MAX_THREADS_PER_BLOCK: 1024
# pycuda._driver.device_attribute.MAX_BLOCK_DIM_X: 1024
# pycuda._driver.device_attribute.MAX_BLOCK_DIM_Y: 1024
# pycuda._driver.device_attribute.MAX_BLOCK_DIM_Z: 64
# pycuda._driver.device_attribute.MAX_GRID_DIM_X: 2147483647
# pycuda._driver.device_attribute.MAX_GRID_DIM_Y: 65535
# pycuda._driver.device_attribute.MAX_GRID_DIM_Z: 65535

module_buff = b"".join(open(os.path.join(cfg.ROOT_DIR, 'src/net/utility', 'remove_empty_box_kernel.cubin'), 'rb').readlines())

def get_gpu_info():
    cuda.init()
    device = cuda.Device(0)
    print(device.get_attributes())

def remove_empty_anchor(view, anchors, limit):
    # input:
    # ahchors: (N, 4) 4->(y1, x1, y2, x2) (x > y)
    # view: (W, H, C) 

    mod = cuda.module_from_buffer(module_buff)
    func = mod.get_function('_Z12remove_emptyPfPiS_S0_S0_')

    anchors_shape = np.array(anchors.shape).astype(np.int32)
    view_shape = np.array(view.shape).astype(np.int32)
    index = np.zeros((anchors.shape[0], view_shape[2])).astype(np.float32)
    func(
        cuda.InOut(index), 
        cuda.In(anchors), 
        cuda.In(view), 
        cuda.In(anchors_shape), 
        cuda.In(view_shape), 
        block=(int(view_shape[2]), 1, 1),  # a thread <-> a value in a specific 2d pos(need to sum the channel)
        grid=(int(anchors_shape[0]), 50, 1)  # a grid <-> an anchor and a line(x)
        # 50 must > anchors width
    )
    index = np.sum(index, axis=1)
    return np.where(index > limit)[0]

if __name__ == '__main__':
    import time
    import random
    import numpy as np
    import pycuda.autoinit
    a = 120000
    anchors = np.hstack([
        0*np.ones((a,1)),
        0*np.ones((a,1)),
        40*np.ones((a,1)),
        16*np.ones((a,1))
    ]).astype(np.int32)
    for i, j in enumerate(anchors):
        l = random.randint(0, 900)
        anchors[i] += l

    view = 0.01*np.random.randn(800, 600, 27).astype(np.float32)
    t = time.time()
    index = remove_empty_anchor(view, anchors, 0)
    print('done, {}'.format(time.time()-t))
    print(index)
    print(len(index))
