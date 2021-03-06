# -*- coding:UTF-8 -*-
#https://github.com/rbgirshick/fast-rcnn/blob/90e75082f087596f28173546cba615d41f0d38fe/lib/fast_rcnn/config.py

"""MV3D config system.

This file specifies default config options for Fast R-CNN. You should not
change values in this file. Instead, you should write a config file (in yaml)
and use cfg_from_file(yaml_file) to load it and override the default options.

Most tools in $ROOT/tools take a --cfg option to specify an override file.
    - See tools/{train,test}_net.py for example code that uses cfg_from_file()
    - See experiments/cfgs/*.yml for example YAML config override files
"""

import os
import os.path as osp
import numpy as np
from time import strftime, localtime
from easydict import EasyDict as edict
import math



__C = edict()
# Consumers can get config by:
#    import config as cfg
cfg = __C
__C.TEST_KEY=11

# for 3dop proposals
__C.LOAD_BEST_PROPOSALS=True
__C.VELODYNE_ANGULAR_RESOLUTION=0.08/180*math.pi 
__C.VELODYNE_VERTICAL_RESOLUTION=0.4/180*math.pi
__C.VELODYNE_HEIGHT=1.73
__C.FRONT_R_OFFSET=70
__C.FRONT_C_OFFSET=750
__C.FRONT_R_MAX=30
__C.FRONT_R_MIN=-70
__C.FRONT_C_MAX=750
__C.FRONT_C_MIN=-750
__C.FRONT_WIDTH=1500
__C.FRONT_HEIGHT=100
__C.BOX3D_Z_MIN=-2.3 # -2.52
__C.BOX3D_Z_MAX=1.5 # -1.02

__C.USE_FRONT=0 #deprecated
__C.USE_TOP_ONLY=1
__C.GPU_AVAILABLE='1'
__C.GPU_USE_COUNT=1
__C.GPU_MEMORY_FRACTION=0.3

# selected object 
__C.DETECT_OBJ=['Car', 'Van']

# for remove empty anchor
__C.ANCHOR_AMOUNT=120000 # 600*800/4/4*4
__C.REMOVE_THRES=0.0

### Hyper-parameters
# for NMS
__C.USE_GPU_NMS=0
#### don't forget to change the TOPN values in configuration.py in the same time!!!
__C.RPN_NMS_THRESHOLD=0.5 # when run, is 0
__C.RCNN_NMS_THRESHOLD=0.4

# for RCNN fusion output(selected result retrieval)
__C.USE_HANDCRAFT_FUSION=0 # use man-restrict fusion rule or just a learnable rule.
__C.HIGH_SCORE_THRESHOLD=0.9
__C.USE_LEARNABLE_FUSION=0

# for Siamese structure context aware refinement
__C.USE_SIAMESE_FUSION=0
__C.ROI_ENLARGE_RATIO=1.5

# for RoI pooling
__C.ROI_POOLING_HEIGHT=6
__C.ROI_POOLING_WIDTH=6

# for using orintation classification
__C.USE_ORINT_CLS=0
__C.ORINT_MAX=90
__C.ORINT_STEP=10 # MAX/STEP=9 classes

# for conv3d on bbox regress
__C.POINT_AMOUNT_LIMIT=100000
__C.VOXEL_ROI_L=0 
__C.VOXEL_ROI_W=0
__C.VOXEL_ROI_H=0
__C.USE_CONV3D=0
__C_USE_POINTNET=0


#['didi2', 'didi','kitti','test']
# 'didi2' means configuration for round 2, 'didi' means configuration for round 1 data, 'kitti' means for kitti dataset.
__C.DATA_SETS_TYPE='kitti'

# Root directory of project
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..'))

if __C.DATA_SETS_TYPE=='test':
    __C.DATA_SETS_DIR = osp.abspath('/home/stu/round12_data_test')
else:
    __C.DATA_SETS_DIR=osp.join(__C.ROOT_DIR, 'data')

__C.RAW_DATA_SETS_DIR = osp.join(__C.DATA_SETS_DIR, 'raw', __C.DATA_SETS_TYPE)
__C.PREPROCESSED_DATA_SETS_DIR = osp.join(__C.DATA_SETS_DIR, 'preprocessed', __C.DATA_SETS_TYPE)
__C.PREPROCESSING_DATA_SETS_DIR = osp.join(__C.DATA_SETS_DIR, 'preprocessing', __C.DATA_SETS_TYPE)
__C.PREDICTED_XML_DIR = osp.join(__C.DATA_SETS_DIR, 'predicted', __C.DATA_SETS_TYPE)

__C.CHECKPOINT_DIR=osp.join(__C.ROOT_DIR,'checkpoint')
__C.LOG_DIR=osp.join(__C.ROOT_DIR,'log')

__C.USE_RESNET_AS_TOP_BASENET = True
__C.USE_RESNET_AS_FRONT_BASENET = True

__C.IMAGE_FUSION_DISABLE = False
__C.RGB_BASENET = 'resnet'  # 'resnet' 、'xception' 'VGG'
if __C.RGB_BASENET == 'xception':
    __C.USE_IMAGENET_PRE_TRAINED_MODEL = True
else:
    __C.USE_IMAGENET_PRE_TRAINED_MODEL =False

__C.TRACKLET_GTBOX_LENGTH_SCALE = 1.6

# image crop config
if __C.DATA_SETS_TYPE ==  'didi' or __C.DATA_SETS_TYPE   ==  'test':
    __C.IMAGE_CROP_LEFT     =0 #pixel
    __C.IMAGE_CROP_RIGHT    =0
    __C.IMAGE_CROP_TOP      =400
    __C.IMAGE_CROP_BOTTOM   =100
elif __C.DATA_SETS_TYPE ==  'didi2':
    __C.IMAGE_CROP_LEFT = 0  # pixel
    __C.IMAGE_CROP_RIGHT = 0
    __C.IMAGE_CROP_TOP = 400
    __C.IMAGE_CROP_BOTTOM = 100
else:
    __C.IMAGE_CROP_LEFT     =0  #pixel
    __C.IMAGE_CROP_RIGHT    =0
    __C.IMAGE_CROP_TOP      =0
    __C.IMAGE_CROP_BOTTOM   =0

# image
if __C.DATA_SETS_TYPE   ==  'test':
    __C.IMAGE_HEIGHT=1096 #pixel
    __C.IMAGE_WIDTH=1368
elif __C.DATA_SETS_TYPE ==  'didi' or __C.DATA_SETS_TYPE ==  'didi2':
    __C.IMAGE_HEIGHT=1096 #pixel
    __C.IMAGE_WIDTH=1368
elif __C.DATA_SETS_TYPE == 'kitti':
    __C.IMAGE_WIDTH=1242
    __C.IMAGE_HEIGHT=375


# config for lidar to top
if __C.DATA_SETS_TYPE == 'didi' or __C.DATA_SETS_TYPE == 'test':
    TOP_Y_MIN = -10
    TOP_Y_MAX = +10
    TOP_X_MIN = -45
    TOP_X_MAX = 45
    TOP_Z_MIN = -3.0
    TOP_Z_MAX = 0.7

    TOP_X_DIVISION = 0.2
    TOP_Y_DIVISION = 0.2
    TOP_Z_DIVISION = 0.3
elif __C.DATA_SETS_TYPE == 'didi2':
    TOP_Y_MIN = -30
    TOP_Y_MAX = 30
    TOP_X_MIN = -50
    TOP_X_MAX = 50
    TOP_Z_MIN = -3.5
    TOP_Z_MAX = 0.6

    TOP_X_DIVISION = 0.2
    TOP_Y_DIVISION = 0.2
    TOP_Z_DIVISION = 0.3
elif __C.DATA_SETS_TYPE == 'kitti':
    TOP_Y_MIN = -30
    TOP_Y_MAX = +30
    TOP_X_MIN = 0
    TOP_X_MAX = 80
    TOP_Z_MIN = -4.2
    TOP_Z_MAX = 0.8

    TOP_X_DIVISION = 0.1
    TOP_Y_DIVISION = 0.1
    TOP_Z_DIVISION = 0.2
else:
    raise ValueError('unexpected type in cfg.DATA_SETS_TYPE item: {}!'.format(__C.DATA_SETS_TYPE))


if __C.DATA_SETS_TYPE == 'kitti':
    __C.MATRIX_Mt = ([[  2.34773698e-04,   1.04494074e-02,   9.99945389e-01,  0.00000000e+00],
                  [ -9.99944155e-01,   1.05653536e-02,   1.24365378e-04,  0.00000000e+00],
                  [ -1.05634778e-02,  -9.99889574e-01,   1.04513030e-02,  0.00000000e+00],
                  [  5.93721868e-02,  -7.51087914e-02,  -2.72132796e-01,  1.00000000e+00]])

    __C.MATRIX_Kt = ([[ 721.5377,    0.    ,    0.    ],
                  [   0.    ,  721.5377,    0.    ],
                  [ 609.5593,  172.854 ,    1.    ]])

    __C.MATRIX_T_VELO_2_CAM = ([
            [ 7.533745e-03 , -9.999714e-01 , -6.166020e-04, -4.069766e-03 ], 
            [ 1.480249e-02 , 7.280733e-04 , -9.998902e-01, -7.631618e-02 ],
            [ 9.998621e-01 , 7.523790e-03 , 1.480755e-02, -2.717806e-01 ],
            [ 0, 0, 0, 1]    
    ])
    __C.MATRIX_R_RECT_0 = ([
            [ 1, 0, 0, 0],
            [ 0, 1, 0, 0],
            [ 0, 0, 1, 0],
            [ 0, 0, 0, 1]
    ])

# if timer is needed.
__C.TRAINING_TIMER = True
__C.TRACKING_TIMER = True
__C.DATAPY_TIMER = False

# print(cfg.RAW_DATA_SETS_DIR)
# print(cfg.PREPROCESSED_DATA_SETS_DIR)
# print(cfg.PREDICTED_XML_DIR)

__C.USE_CLIDAR_TO_TOP = False

def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.iteritems():
        # a must specify keys that are in b
        if not b.has_key(k):
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                'for config key: {}').format(type(b[k]),
                                                            type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v

def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)

def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert d.has_key(subkey)
            d = d[subkey]
        subkey = key_list[-1]
        assert d.has_key(subkey)
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(
            type(value), type(d[subkey]))
        d[subkey] = value

if __name__ == '__main__':
    print('__C.ROOT_DIR = '+__C.ROOT_DIR)
    print('__C.DATA_SETS_DIR = '+__C.DATA_SETS_DIR)
    print('__C.RAW_DATA_SETS_DIR = '+__C.RAW_DATA_SETS_DIR)
