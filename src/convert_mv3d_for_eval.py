# -*- coding:utf-8 -*-
#!/bin/env python3.5

import glob
import numpy as np 
import os 
from config import *
from multiprocessing import Pool
import data
import cv2

import net.utility.draw as nud

base_res_path = '/home/mxj/workspace_mxj/eval-kitti/MV3D_dev/mv3d_test'
f_boxes3d = glob.glob(os.path.join(base_res_path, '*_boxes3d.npy'))
f_boxes3d.sort()
f_label = glob.glob(os.path.join(base_res_path, '*_labels.npy'))
f_label.sort()
f_probs = glob.glob(os.path.join(base_res_path, '*_probs.npy'))
f_probs.sort()
base_dataset_path = '/data/mxj/kitti/object_3dop/training'
f_top_view = glob.glob(os.path.join(base_dataset_path, 'top_view', '*.npy'))
f_top_view.sort() 
f_top_view = f_top_view[1:]
f_rgb = glob.glob(os.path.join(base_dataset_path, 'image_2', '*.png'))
f_rgb.sort()
f_rgb = f_rgb[1:]
tags = [name.split('/')[-1].split('.')[-2] for name in f_rgb]
assert(len(f_boxes3d) == len(f_label) == len(f_probs) == len(f_top_view) == len(f_rgb))

def generate_result(tag, boxes3d, prob):
	line = "{} "
	for box, p in zip(boxes3d, prob)



def handler(i):
	tag = tags[i]
	top = np.load(f_top_view[i])
	rgb = cv2.imread(f_rgb[i])
	boxes3d = np.load(f_boxes3d[i])
	prob = np.load(f_probs[i])
	predict_log(tag, top, rgb, boxes3d, prob)
	generate_result(tag, boxes3d, prob)
	print(i)

def main():
	pro = Pool(10)
	pro.map(handler, [i for i in range(len(f_top_view))])

def top_image_padding(top_image):
    return np.concatenate((top_image, np.zeros_like(top_image)*255,np.zeros_like(top_image)*255), 1)

def predict_log(tag, top_view, rgb, boxes3d, probs, gt_boxes3d=[]):
	top_image = data.draw_top_image(top_view)
	top_image = top_image_padding(top_image)

	text_lables = ['No.%d class:1 prob: %.4f' % (i, prob) for i, prob in enumerate(probs)]
	predict_camera_view = nud.draw_box3d_on_camera(rgb, boxes3d, text_lables=text_lables)

	predict_top_view = data.draw_box3d_on_top(top_image, boxes3d)

    # draw gt on camera and top view:
	if len(gt_boxes3d) > 0: # array size > 1 cannot directly used in if
		predict_top_view = data.draw_box3d_on_top(predict_top_view, gt_boxes3d, color=(0, 0, 255))
		predict_camera_view = draw_box3d_on_camera(predict_camera_view, gt_boxes3d, color=(0, 0, 255))

	new_size = (predict_camera_view.shape[1] // 2, predict_camera_view.shape[0] // 2)
	predict_camera_view = cv2.resize(predict_camera_view, new_size)
	cv2.imwrite(os.path.join('/data/mxj/kitti/mv3d_result/image', tag + 'rgb_.png'), predict_camera_view)
	cv2.imwrite(os.path.join('/data/mxj/kitti/mv3d_result/image', tag + 'top_.png'), predict_top_view)


if __name__ == '__main__':
	main()