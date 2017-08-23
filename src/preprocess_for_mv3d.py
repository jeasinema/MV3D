# -*- coding:utf-8 -*-
#!/bin/env python3.5

import glob
import numpy as np 
import data 
import cv2
from config import * 
from raw_data import Lidar
from multiprocessing import Pool
import os

dataset = {
            '2011_09_26': ['0001', '0002', '0005', '0011', '0013', '0015', '0017', '0018',  '0019', '0020', '0023',
                       '0027', '0028', '0029', '0035', '0036', '0039', '0046', '0048', '0051', '0052', '0056', '0057', '0059',
                           '0060', '0061', '0064', '0070', '0079', '0084', '0086', '0091']
        }

p = data.Preprocess()
lidar = Lidar(dataset)


def handler(tag):
	scan = lidar.load(tag)
	top = data.lidar_to_top(scan)
	front = p.lidar_to_front_fast(scan)
	os.makedirs(os.path.join(cfg.RAW_DATA_SETS_DIR, 'top_view', *tag.split('/')[:-1]), exist_ok=True)
	os.makedirs(os.path.join(cfg.RAW_DATA_SETS_DIR, 'front_view', *tag.split('/')[:-1]), exist_ok=True)
	np.save(os.path.join(cfg.RAW_DATA_SETS_DIR, 'top_view', tag + '.npy'), top)
	np.save(os.path.join(cfg.RAW_DATA_SETS_DIR, 'front_view', tag + '.npy'), front)
	print(tag)

def main():
	tags = lidar.get_paths_mapping()
	pro = Pool(12)
	pro.map(handler, tags)



if __name__ == '__main__':
	main()