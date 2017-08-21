# -*- coding: utf-8 -*-

import cv2
import numpy as np
from config import cfg


def convert_points_to_croped_image(img_points):
    img_points = img_points.copy()

    left = cfg.IMAGE_CROP_LEFT  # pixel
    right = cfg.IMAGE_CROP_RIGHT
    top = cfg.IMAGE_CROP_TOP
    bottom = cfg.IMAGE_CROP_BOTTOM

    croped_img_h = proj.image_height - top - bottom
    croped_img_w = proj.image_width - left - right

    img_points[:, 1] -= top
    mask = img_points[:, 1] < 0
    img_points[mask, 1] = 0
    out_range_mask = mask

    mask = img_points[:, 1] >= croped_img_h
    img_points[mask, 1] = croped_img_h - 1
    out_range_mask = np.logical_or(out_range_mask, mask)

    img_points[:, 0] -= left
    mask = img_points[:, 0] < 0
    img_points[mask, 0] = 0
    out_range_mask = np.logical_or(out_range_mask, mask)

    mask = img_points[:, 0] >= croped_img_w
    img_points[mask, 0] = croped_img_w - 1
    out_range_mask = np.logical_or(out_range_mask, mask)

    return img_points, out_range_mask


def box3d_to_rgb_projection_cv2(points):
    # http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html

    # cameraMatrix=np.array([[1384.621562, 0.000000, 625.888005],
    #                          [0.000000, 1393.652271, 559.626310],
    #                          [0.000000, 0.000000, 1.000000]])

    # https://github.com/zxf8665905/lidar-camera-calibration/blob/master/Calibration.ipynb
    # Out[17]:
    # x=np.array([ -1.50231172e-03,  -4.00842946e-01,  -5.30289086e-01,
    #    -2.41054475e+00,   2.41781181e+00,  -2.46716659e+00])
    #tx, ty, tz, rx, ry, rz = x

    #rotVect = np.array([rx, ry, rz])
    #transVect = np.array([tx, ty, tz])

    #distCoeffs=np.array([[-0.152089, 0.270168, 0.003143, -0.005640, 0.000000]])

    #imagePoints, jacobia=cv2.projectPoints(points,rotVect,transVect,cameraMatrix,distCoeffs)
    # imagePoints=np.reshape(imagePoints,(8,2))

    projMat = np.matrix([[6.24391515e+02,  -1.35999541e+03,  -3.47685065e+01,  -8.19238784e+02],
                         [5.20528665e+02,   1.80893752e+01,  -
                             1.38839738e+03,  -1.17506110e+03],
                         [9.99547104e-01,   3.36246424e-03,  -2.99045429e-02,  -1.34871685e+00]])
    imagePoints = []
    for pt in points:
        X = projMat * np.matrix(list(pt) + [1]).T
        X = np.array(X[:2, 0] / X[2, 0]).flatten()
        imagePoints.append(X)
    imagePoints = np.array(imagePoints)

    return imagePoints.astype(np.int)


def draw_rgb_projections(image, projections, color=(255, 0, 255), thickness=2, darker=1.0):

    img = (image.copy() * darker).astype(np.uint8)
    num = len(projections)
    for n in range(num):
        qs = projections[n]
        for k in range(0, 4):
            # http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
            i, j = k, (k + 1) % 4
            cv2.line(img, (qs[i, 0], qs[i, 1]), (qs[j, 0],
                                                 qs[j, 1]), color, thickness, cv2.LINE_AA)

            i, j = k + 4, (k + 1) % 4 + 4
            cv2.line(img, (qs[i, 0], qs[i, 1]), (qs[j, 0],
                                                 qs[j, 1]), color, thickness, cv2.LINE_AA)

            i, j = k, k + 4
            cv2.line(img, (qs[i, 0], qs[i, 1]), (qs[j, 0],
                                                 qs[j, 1]), color, thickness, cv2.LINE_AA)

    return img


def box3d_to_rgb_box(boxes3d, Mt=None, Kt=None):
    if (True):
        if Mt is None:
            Mt = np.array(cfg.MATRIX_Mt)
        if Kt is None:
            Kt = np.array(cfg.MATRIX_Kt)

        num = len(boxes3d)
        projections = np.zeros((num, 8, 2),  dtype=np.int32)
        for n in range(num):
            box3d = boxes3d[n]
            Ps = np.hstack((box3d, np.ones((8, 1))))
            Qs = np.matmul(Ps, Mt)
            Qs = Qs[:, 0:3]
            qs = np.matmul(Qs, Kt)
            zs = qs[:, 2].reshape(8, 1)
            qs = (qs / zs)
            projections[n] = qs[:, 0:2]
        return projections

    else:
        num = len(boxes3d)
        projections = np.zeros((num, 8, 2), dtype=np.int32)
        for n in range(num):
            box3d = boxes3d[n].copy()
            if np.sum(box3d[:, 0] > 0) > 0:
                box2d = box3d_to_rgb_projection_cv2(box3d)
                box2d, out_range = convert_points_to_croped_image(box2d)
                if np.sum(out_range == False) >= 2:
                    projections[n] = box2d
        return projections


def draw_box3d_on_camera(rgb, boxes3d, color=(255, 0, 255), thickness=1, text_lables=[]):
    projections = box3d_to_rgb_box(boxes3d)
    rgb = draw_rgb_projections(
        rgb, projections, color=color, thickness=thickness)
    font = cv2.FONT_HERSHEY_SIMPLEX
    return rgb


def draw_fusion_target():
    boxes3d = np.load('/home/mxj/boxes3d.npy')
    labels = np.load('/home/mxj/labels.npy')
    cam_img = np.load('/home/mxj/cam_raw.npy')
    class_color = [(10, 20, 10), (255, 0, 0)]
    for i, label in enumerate(labels):
        color = class_color[label]
        cam_img = draw_box3d_on_camera(
            cam_img, boxes3d[i:i + 1, :, :], (color[0], color[1], color[2]))
    return cam_img


def main():
    cv2.imwrite('1.png', draw_fusion_target())


if __name__ == '__main__':
    main()
