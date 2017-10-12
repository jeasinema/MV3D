# -*- coding:utf-8 -*-
#!/bin/env python3.5

import glob
import numpy as np
import math
import os
from config import *
from multiprocessing import Pool
import data
import cv2
import argparse

import net.utility.draw as nud
import net.processing.boxes3d as box


parser = argparse.ArgumentParser(description='testing')

parser.add_argument('-t', '--target-dir', type=str, nargs='?', required=True)
parser.add_argument('-s', '--source-dir', type=str, nargs='?', required=True)
parser.add_argument('-k', '--top-k', type=int, nargs='?', default=10)

args = parser.parse_args()

top_k = args.top_k

print('\n\n{}\n\n'.format(args))

os.makedirs(os.path.join(args.target_dir, 'result'), exist_ok=True)
os.makedirs(os.path.join(args.target_dir, 'image'), exist_ok=True)

base_res_path = args.source_dir
f_boxes3d = glob.glob(os.path.join(base_res_path, '*_boxes3d.npy'))
f_boxes3d.sort()
f_label = glob.glob(os.path.join(base_res_path, '*_labels.npy'))
f_label.sort()
f_probs = glob.glob(os.path.join(base_res_path, '*_probs.npy'))
f_probs.sort()
base_dataset_path = '/home/maxiaojian/data/kitti/object/training'
f_top_view = glob.glob(os.path.join(base_dataset_path, 'top_view', '*.npy'))
f_top_view.sort()
f_top_view = f_top_view
f_rgb = glob.glob(os.path.join(base_dataset_path, 'image_2', '*.png'))
f_rgb.sort()
f_rgb = f_rgb
tags = [name.split('/')[-1].split('.')[-2] for name in f_rgb]
# assert(len(f_boxes3d) == len(f_label) == len(f_probs) == len(f_top_view) == len(f_rgb))
assert(len(f_boxes3d) == len(f_probs))

# return 3d projections


def box3d_to_rgb_box(boxes3d, Mt=None, Kt=None):
    if Mt is None:
        Mt = np.array(cfg.MATRIX_Mt)
    if Kt is None:
        Kt = np.array(cfg.MATRIX_Kt)

    num = len(boxes3d)
    projections = np.zeros((num, 8, 3), dtype=np.float32)
    for n in range(num):
        box3d = boxes3d[n]
        Ps = np.hstack((box3d, np.ones((8, 1))))
        Qs = np.matmul(Ps, Mt)
        Qs = Qs[:, 0:3]
        qs = np.matmul(Qs, Kt)
        zs = qs[:, 2].reshape(8, 1)
        qs = (qs / zs)
        projections[n] = qs
    return projections


def project_to_rgb_roi(rois3d):
    num = len(rois3d)
    rois = np.zeros((num, 5), dtype=np.int32)
    projections = box3d_to_rgb_box(rois3d)
    for n in range(num):
        qs = projections[n]
        minx = int(np.min(qs[:, 0]))
        maxx = int(np.max(qs[:, 0]))
        miny = int(np.min(qs[:, 1]))
        maxy = int(np.max(qs[:, 1]))
        rois[n, 1:5] = minx, miny, maxx, maxy

    return rois, projections


def generate_result(tag, boxes3d, prob):
    def corner2center(rois3d):
        ret = []
        for roi in rois3d:
            if 1:  # average version
                roi = np.array(roi)
                h = abs(np.sum(roi[:4, 1] - roi[4:, 1]) / 4)
                w = np.sum(
                    np.sqrt(np.sum((roi[0, [0, 2]] - roi[3, [0, 2]])**2)) +
                    np.sqrt(np.sum((roi[1, [0, 2]] - roi[2, [0, 2]])**2)) +
                    np.sqrt(np.sum((roi[4, [0, 2]] - roi[7, [0, 2]])**2)) +
                    np.sqrt(np.sum((roi[5, [0, 2]] - roi[6, [0, 2]])**2))
                ) / 4
                l = np.sum(
                    np.sqrt(np.sum((roi[0, [0, 2]] - roi[1, [0, 2]])**2)) +
                    np.sqrt(np.sum((roi[2, [0, 2]] - roi[3, [0, 2]])**2)) +
                    np.sqrt(np.sum((roi[4, [0, 2]] - roi[5, [0, 2]])**2)) +
                    np.sqrt(np.sum((roi[6, [0, 2]] - roi[7, [0, 2]])**2))
                ) / 4
                x, y, z = np.sum(roi, axis=0) / 8
                ry = np.sum(
                    math.atan2(roi[2, 0] - roi[1, 0], roi[2, 2] - roi[1, 2]) +
                    math.atan2(roi[6, 0] - roi[5, 0], roi[6, 2] - roi[5, 2]) +
                    math.atan2(roi[3, 0] - roi[0, 0], roi[3, 2] - roi[0, 2]) +
                    math.atan2(roi[7, 0] - roi[4, 0], roi[7, 2] - roi[4, 2]) +
                    math.atan2(roi[0, 2] - roi[1, 2], roi[1, 0] - roi[0, 0]) +
                    math.atan2(roi[4, 2] - roi[5, 2], roi[5, 0] - roi[4, 0]) +
                    math.atan2(roi[3, 2] - roi[2, 2], roi[2, 0] - roi[3, 0]) +
                    math.atan2(roi[7, 2] - roi[6, 2], roi[6, 0] - roi[7, 0])
                ) / 8
            else:  # max version
                h = max(abs(roi[:4, 1] - roi[4:, 1]))
                w = np.max(
                    np.sqrt(np.sum((roi[0, [0, 2]] - roi[3, [0, 2]])**2)) +
                    np.sqrt(np.sum((roi[1, [0, 2]] - roi[2, [0, 2]])**2)) +
                    np.sqrt(np.sum((roi[4, [0, 2]] - roi[7, [0, 2]])**2)) +
                    np.sqrt(np.sum((roi[5, [0, 2]] - roi[6, [0, 2]])**2))
                )
                l = np.max(
                    np.sqrt(np.sum((roi[0, [0, 2]] - roi[1, [0, 2]])**2)) +
                    np.sqrt(np.sum((roi[2, [0, 2]] - roi[3, [0, 2]])**2)) +
                    np.sqrt(np.sum((roi[4, [0, 2]] - roi[5, [0, 2]])**2)) +
                    np.sqrt(np.sum((roi[6, [0, 2]] - roi[7, [0, 2]])**2))
                )
                x, y, z = np.sum(roi, axis=0) / 8
                ry = np.sum(
                    math.atan2(roi[2, 0] - roi[1, 0], roi[2, 2] - roi[1, 2]) +
                    math.atan2(roi[6, 0] - roi[5, 0], roi[6, 2] - roi[5, 2]) +
                    math.atan2(roi[3, 0] - roi[0, 0], roi[3, 2] - roi[0, 2]) +
                    math.atan2(roi[7, 0] - roi[4, 0], roi[7, 2] - roi[4, 2]) +
                    math.atan2(roi[0, 2] - roi[1, 2], roi[1, 0] - roi[0, 0]) +
                    math.atan2(roi[4, 2] - roi[5, 2], roi[5, 0] - roi[4, 0]) +
                    math.atan2(roi[3, 2] - roi[2, 2], roi[2, 0] - roi[3, 0]) +
                    math.atan2(roi[7, 2] - roi[6, 2], roi[6, 0] - roi[7, 0])
                ) / 8
            ret.append([h, w, l, x, y, z, ry])
        return ret

    line = "Car 0 0 0 {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}\n"
    rgb_rois2d, _ = project_to_rgb_roi(boxes3d)
    rgb_rois3d = box.box3d_to_camera_box3d(boxes3d)
    rgb_rois3d = np.array(corner2center(rgb_rois3d))  # h, w, l, x, y, z, ry
    with open(os.path.join(args.target_dir, 'result', tag + '.txt'), 'w+') as of:
        for box2d, box3d, p in zip(rgb_rois2d, rgb_rois3d, prob):
            if p > 0:  # FIXME
                res = line.format(*box2d[1:5], *box3d, p)
                of.write(res)


def handler(i):
    global top_k
    tag = tags[i]
    # top = np.load(f_top_view[i])
    # rgb = cv2.imread(f_rgb[i])
    boxes3d = np.load(f_boxes3d[i])[:top_k]
    prob = np.load(f_probs[i])[:top_k]
    # visualize_result(tag, top, rgb, boxes3d, prob)
    generate_result(tag, boxes3d, prob)
    print(i)


def main(args):
    pro = Pool(10)
    # handler(3000)
    pro.map(handler, [i for i in range(len(f_boxes3d))])


def top_image_padding(top_image):
    return np.concatenate((top_image, np.zeros_like(top_image) * 255, np.zeros_like(top_image) * 255), 1)


def visualize_result(tag, top_view, rgb, boxes3d, probs, gt_boxes3d=[]):
    top_image = data.draw_top_image(top_view)
    top_image = top_image_padding(top_image)

    text_lables = ['No.%d class:1 prob: %.4f' %
                   (i, prob) for i, prob in enumerate(probs)]
    predict_camera_view = nud.draw_box3d_on_camera(
        rgb, boxes3d, text_lables=text_lables)

    predict_top_view = data.draw_box3d_on_top(top_image, boxes3d)

    # draw gt on camera and top view:
    if len(gt_boxes3d) > 0:  # array size > 1 cannot directly used in if
        predict_top_view = data.draw_box3d_on_top(
            predict_top_view, gt_boxes3d, color=(0, 0, 255))
        predict_camera_view = draw_box3d_on_camera(
            predict_camera_view, gt_boxes3d, color=(0, 0, 255))

    new_size = (predict_camera_view.shape[1] //
                2, predict_camera_view.shape[0] // 2)
    predict_camera_view = cv2.resize(predict_camera_view, new_size)
    cv2.imwrite(os.path.join(args.target_dir, 'image',
                             tag + 'rgb_.png'), predict_camera_view)
    cv2.imwrite(os.path.join(args.target_dir, 'image',
                             tag + 'top_.png'), predict_top_view)


if __name__ == '__main__':
    main(args)
