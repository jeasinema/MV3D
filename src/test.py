import mv3d
import mv3d_net
import glob
from config import *
import utils.batch_loading as ub
import argparse
import math
import os
import sys
import time
from utils.training_validation_data_splitter import TrainingValDataSplitter
from utils.batch_loading import Loading3DOP, KittiLoading
from utils.batch_loading import BatchLoading3
import net.processing.boxes3d  as box
import data as Data
from net.rpn_target_op import make_bases
from net.rpn_nms_op     import draw_rpn_proposal
import numpy as np


def test_3dop(args):
    # 3dop_proposal should be (N, 8) (7 for pos, 1 for objectness score)
    with Loading3DOP(object_dir='~/data/kitti/object_3dop', proposals_dir='/data/mxj/kitti/3dop_proposal', queue_size=20, require_shuffle=False, 
         is_testset=True) as testset:
      test = mv3d.Tester_3DOP(*testset.get_shape(), log_tag=args.tag)
      data = testset.load()
      count = 0
      while data:
        boxes, labels = test(*data)
        data = testset.load()
        print("Process {} data".format(count))
        boxes, labels = np.array(boxes), np.array(labels)
        np.save(os.path.join(args.target_dir, str(count) + "_boxes.npy"), boxes)
        np.save(os.path.join(args.target_dir, str(count) + "_labels.npy"), labels)

        count += 1

def test_rpn(args):
  with KittiLoading(object_dir='~/data/kitti/object', queue_size=50, require_shuffle=False,
    is_testset=True, use_precal_view=False) as testset:
    os.makedirs(args.target_dir, exist_ok=True)
    test = mv3d.Tester_RPN(*testset.get_shape(),log_tag=args.tag)
    data = testset.load()
    count = 1
    while data:
      print('Process {} data'.format(count))
      tag, rgb, _, top_view, front_view = data
      box3d, rgb_roi, top_roi, roi_score = test(top_view, front_view, rgb)
      # ret = np.hstack((box3d, roi_score))
      test.log_rpn(step=0, scope_name='test_rpn')
      np.save(os.path.join(args.target_dir, '{}_boxes3d.npy'.format(tag[0])), box3d)
      np.save(os.path.join(args.target_dir, '{}_boxes_top.npy'.format(tag[0])), box3d)
      np.save(os.path.join(args.target_dir, '{}_score.npy'.format(tag[0])), roi_score)
      img = draw_rpn_proposal(test.top_image, top_roi, roi_score)
      test.summary_image(img, 'test_rpn' + '/img_rpn_proposal',step=0) # just all the proposals. after nms, lighter color, higher score
      img_gt = draw_rpn_gt(test.top_image, test.batch_gt_top_boxes, test.batch_gt_labels)
      test.summary_image(img_gt, 'test_rpn' + '/img_rpn_gt',step=0) # just all the proposals. after nms, lighter color, higher score

      data = testset.load()
      count += 1

def test_rpn_raw_dataset(args):
    with BatchLoading3(tags=training_dataset, require_shuffle=True, use_precal_view=False, queue_size=5, use_multi_process_num=1) as testset:
        count = 0
        test = mv3d.Tester_RPN(*testset.get_shape(),log_tag=args.tag)
        print('end init')
        topk = 100
        repeat = False
        while True:
            if repeat == False:
                test.batch_rgb_images, test.batch_top_view, test.batch_front_view, \
                test.batch_gt_labels, test.batch_gt_boxes3d, test.frame_id = \
                    testset.load()
                print('done load')
                print(test.batch_gt_labels)
                print(test.batch_gt_boxes3d)
                print(test.frame_id)
            box3d, rgb_roi, top_roi, roi_score = test(test.batch_top_view, test.batch_front_view, test.batch_rgb_images)
            print('done test')
            index = np.argsort(roi_score)[::-1][:topk]
            test.batch_proposals = test.batch_proposals[index]
            test.batch_proposal_scores = test.batch_proposal_scores[index]
            print(test.batch_proposal_scores)
            test.log_rpn(step=count, scope_name='test_rpn')
            count += 1
            print('done log')
            print('input anything for continue.')
            index = sys.stdin.readline()[:-1]
            if index == 'n':
                repeat = False
                continue
            elif index == 'q':
                sys.exit(0)
            else:
                try:
                    topk = int(index)
                    repeat = True
                except:
                    print('invalid input')
                    repeat = True
                    continue
                

def test_rpn_origin_dataset(args):
    with KittiLoading(object_dir='/home/maxiaojian/data/kitti/object', queue_size=50, require_shuffle=False, 
         is_testset=False, use_precal_view=False, use_multi_process_num=0) as testset:
        count = 0
        test = mv3d.Tester_RPN(*testset.get_shape(),log_tag=args.tag)
        print('end init')
        topk = 100
        index = 1
        while True:
            data = testset.load_specified(index)
            tag, label, rgb, _, top_view, front_view = data 
            test.batch_rgb_images = rgb 
            test.batch_top_view = top_view 
            test.batch_front_view = front_view 
            test.batch_gt_boxes3d, test.batch_gt_labels = Data.kitti_label_to_lidar_box3d(label[0], 'Car', positive_only=False)
            # test.batch_gt_labels = np.ones((1, test.batch_gt_boxes3d.shape[1]))
            test.frame_id = tag 
            print('done load')
            print(test.frame_id)
            box3d, rgb_roi, top_roi, roi_score, top_rpn_heatmap = test(test.batch_top_view, test.batch_front_view, test.batch_rgb_images)
            print('done test')
            index = np.argsort(roi_score)[::-1][:topk]
            test.batch_proposals = test.batch_proposals[index]
            test.batch_proposal_scores = test.batch_proposal_scores[index]
            print(test.batch_proposal_scores)
            test.log_rpn(step=count, scope_name='test_rpn')
            
            # generate heatmap 
            # top_rpn_heatmap: (1, W, H, 4*2)
            top_rpn_heatmap = top_rpn_heatmap[0, :, :, 1::2]
            heatmap = np.max(top_rpn_heatmap, axis=2)
            heatmap = heatmap.clip(min=-400, max=10)
            heatmap = (1./(1.+(math.e**(-heatmap)))*255).astype(np.int32)
            from IPython import embed; embed()
            img = np.dstack([heatmap, heatmap, heatmap])
            test.summary_image(img, 'test_rpn' + '/rpn_cls_heatmap', step=0)  #just RPN gt
            count += 1
            print('done log')
            print('input anything for continue.')
            indata = sys.stdin.readline()[:-1]
            if indata == 'n':
                continue
            elif indata == 'q':
                sys.exit(0)
            else:
                try:
                    index, topk = [int(i) for i in indata.split()]
                except:
                    print('invalid input')
                    continue

 def test_rpn_origin_dataset_batch(args):
    with KittiLoading(object_dir='/home/maxiaojian/data/kitti/object', queue_size=50, require_shuffle=False, 
         is_testset=False, use_precal_view=False, use_multi_process_num=0) as testset:
        count = 0
        test = mv3d.Tester_RPN(*testset.get_shape(),log_tag=args.tag)
        print('end init')
        topk = 100
        index = 0
        for index in range(len(testset)):
            data = testset.load_specipied(index)
            tag, label, rgb, _, top_view, front_view = data 
            test.batch_rgb_images = rgb 
            test.batch_top_view = top_view 
            test.batch_front_view = front_view 
            test.batch_gt_boxes3d, test.batch_gt_labels = Data.kitti_label_to_lidar_box3d(label[0], 'Car', positive_only=False)
            # test.batch_gt_labels = np.ones((1, test.batch_gt_boxes3d.shape[1]))
            test.frame_id = tag 
            print('done load')
            print(test.frame_id)
            box3d, rgb_roi, top_roi, roi_score, top_rpn_heatmap = test(test.batch_top_view, test.batch_front_view, test.batch_rgb_images)
            print('done test')
            index = np.argsort(roi_score)[::-1][:topk]
            test.batch_proposals = test.batch_proposals[index]
            test.batch_proposal_scores = test.batch_proposal_scores[index]
            print(test.batch_proposal_scores)
            test.log_rpn(step=count, scope_name='test_rpn')

            # eval all file 
               

def test_mv3d(args):
    with KittiLoading(object_dir='~/data/kitti/object', queue_size=50, require_shuffle=False,
        is_testset=True, use_precal_view=True) as testset:
        os.makedirs(args.target_dir, exist_ok=True)
        test = mv3d.Predictor(*testset.get_shape(), log_tag=args.tag)
        data = testset.load()

        count = 1
        while data:
            print('Process {} data'.format(count))
            tag, rgb, _, top_view, front_view = data 
            boxes3d, labels, probs = test(top_view, front_view, rgb, score_threshold=0.5)
            np.save(os.path.join(args.target_dir, '{}_boxes3d.npy'.format(tag[0])), boxes3d)
            np.save(os.path.join(args.target_dir, '{}_labels.npy'.format(tag[0])), labels)        
            np.save(os.path.join(args.target_dir, '{}_probs.npy'.format(tag[0])), probs)        

            data = testset.load()
            count += 1

def test_single_mv3d(args):
    with KittiLoading(object_dir='/home/maxiaojian/data/kitti/object', queue_size=1, require_shuffle=False,
        is_testset=False, use_precal_view=True) as testset:
        test = mv3d.Predictor_for_test(*testset.get_shape(), log_tag=args.tag)
        while True:
            print('Please input frame index for test, q for exit:')
            index = sys.stdin.readline()
            if index == 'q\n': sys.exit()
            index, threshold = index.split(',')
            index = int(index)
            threshold = float(threshold)
            try:
              data = testset.load_specified(index)
              tag, label, rgb, _, top_view, front_view = data 
              gt_box3d = Data.kitti_label_to_lidar_box3d(label[0], 'Car', positive_only=True)
              boxes3d, labels, probs = test(top_view, front_view, rgb, threshold, gt_boxes3d=gt_box3d)
              test.dump_log(args.target_dir, index)
            except:
              print('test {} failed!'.format(index))

def test_rpn_target_interact(args):
    """ mainly for testing the anchor design """
    ratios=np.array([0.5,1,2], dtype=np.float32)
    scales=np.array([1,2,3],   dtype=np.float32)
    bases = make_bases(
        base_size = 16,
        ratios=ratios,  #aspet ratio
        scales=scales
    )
    # bases = np.array([
    #     [4.5, 2.5, 10.5, 12.5],# (1.0, 0.6)
    #     [2.5, 4.5, 12.5, 10.5],
    #     [-0.5, -12, 15.5, 27], #(3.9, 1.6)
    #     [-12, -0.5, 27, 15.5]
    # ])

    with KittiLoading(object_dir='~/data/kitti/object', queue_size=1, require_shuffle=False,
        is_testset=False, use_precal_view=True) as testset:
        test = mv3d.Tester_RPN_Target(*testset.get_shape(), log_tag=args.tag)
        while True:
            print('Please input frame index for test, q for exit:')
            index = sys.stdin.readline()
            if index == 'q\n': sys.exit()
            index = int(index)
            try:
              data = testset.load_specified(index)
              tag, label, rgb, _, top_view, front_view = data 
              gt_box3d = Data.kitti_label_to_lidar_box3d(label[0], 'Car', positive_only=True)  # only support batch size  == 1
              gt_labels = np.ones((1, len(gt_box3d[0]))) # only support batch size == 1
              amount, amount_pos = test(top_view, front_view, rgb, bases, gt_boxes3d=gt_box3d, gt_labels=gt_labels)
              print('anchor amount: {}, pos:{}'.format(amount, amount_pos))
            except:
              print('test {} failed!'.format(index))

def test_rpn_target(args):
    """ mainly for testing the anchor design """
    ratios=np.array([0.5,1,2], dtype=np.float32)
    scales=np.array([1,2,3],   dtype=np.float32)
    bases = make_bases(
        base_size = 16,
        ratios=ratios,  #aspet ratio
        scales=scales
    )
    # bases = np.array([
    #     [4.5, 2.5, 10.5, 12.5],# (1.0, 0.6)
    #     [2.5, 4.5, 12.5, 10.5],
    #     [-0.5, -12, 15.5, 27], #(3.9, 1.6)
    #     [-12, -0.5, 27, 15.5]
    # ])

    with KittiLoading(object_dir='~/data/kitti/object', queue_size=1, require_shuffle=False,
        is_testset=False, use_precal_view=True) as testset:
        test = mv3d.Tester_RPN_Target(*testset.get_shape(), log_tag=args.tag)
        res = []
        length = len(testset) if len(testset) < 1000 else 1000
        for index in range(length):
            try:
              data = testset.load_specified(index)
              tag, label, rgb, _, top_view, front_view = data 
              gt_box3d = Data.kitti_label_to_lidar_box3d(label[0], 'Car', positive_only=True)  # only support batch size  == 1
              gt_labels = np.ones((1, len(gt_box3d[0]))) # only support batch size == 1
              amount, amount_pos = test(top_view, front_view, rgb, bases, gt_boxes3d=gt_box3d, gt_labels=gt_labels)
              print('{} anchor amount: {}, pos:{}'.format(index, amount, amount_pos))
              res.append((amount, amount_pos))
            except:
              print('test {} failed!'.format(index))
        np.save(args.save_name + '.npy', np.array(res))


def test_front(args):
  pass

def lidar_to_front(lidar):

    def cal_height(point):
        return np.clip(point[2] + cfg.VELODYNE_HEIGHT, a_min=0, a_max=None)
    def cal_distance(point):
        return math.sqrt(sum(np.array(point)**2))
    def cal_intensity(point):
        return point[3]
    def to_front(point):
        return (
            int(math.atan2(point[1], point[0])/cfg.VELODYNE_ANGULAR_RESOLUTION),
            int(math.atan2(point[2], math.sqrt(point[0]**2 + point[1]**2)) \
                /cfg.VELODYNE_VERTICAL_RESOLUTION)
        )

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

    r, c, l = [], [], []
    for point in lidar:
        pc, pr = to_front(point)
        if cfg.FRONT_C_MIN < pc < cfg.FRONT_C_MAX and cfg.FRONT_R_MIN < pr < cfg.FRONT_R_MAX: 
            c.append(pc)
            r.append(pr)
            l.append(point)
    c, r = np.array(c).astype(np.int32), np.array(r).astype(np.int32)
    c += int(cfg.FRONT_C_OFFSET)
    r += int(cfg.FRONT_R_OFFSET)

    ## FIXME simply /2 for resize
    #c //= 2
    #r //= 2

    channel = 3 # height, distance, intencity
    front = np.zeros((cfg.FRONT_WIDTH, cfg.FRONT_HEIGHT, channel+1), dtype=np.float32)
    for point, p_c, p_r in zip(l, c, r):
        if 0 <= p_c < cfg.FRONT_WIDTH and 0 <= p_r < cfg.FRONT_HEIGHT:
            front[p_c, p_r, 0:channel] *= front[p_c, p_r, channel]
            front[p_c, p_r, 0:channel] += np.array([cal_height(point), cal_distance(point), cal_intensity(point)])
            front[p_c, p_r, channel] += 1
            front[p_c, p_r, 0:channel] /= front[p_c, p_r, channel]

    return front[:, :, 0:channel]

def lidar_to_front_fast(lidar):

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

    channel = 3 # height, distance, intencity
    front = np.zeros((cfg.FRONT_WIDTH, cfg.FRONT_HEIGHT, channel), dtype=np.float32)
    weight_mask = np.zeros_like(front[:, :, 0])
    def _add(x):
        weight_mask[int(x[0]), int(x[1])] += 1
    def _fill(x):
        front[int(x[0]), int(x[1]), :] += x[2:]
    np.apply_along_axis(_add, 1, points)
    weight_mask[weight_mask == 0] = 1
    buf = np.hstack((points, cal_height(lidar), cal_distance(lidar), cal_intensity(lidar)))
    np.apply_along_axis(_fill, 1, buf)
    front /= weight_mask[:, :, np.newaxis]

    return front
def lidar_to_top(lidar):

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

    if (cfg.DATA_SETS_TYPE == 'didi' or cfg.DATA_SETS_TYPE == 'test' or cfg.DATA_SETS_TYPE=='didi2'):
        lidar=filter_center_car(lidar)


    pxs=lidar[:,0]
    pys=lidar[:,1]
    pzs=lidar[:,2]
    prs=lidar[:,3]
    qxs=((pxs-TOP_X_MIN)//TOP_X_DIVISION).astype(np.int32)
    qys=((pys-TOP_Y_MIN)//TOP_Y_DIVISION).astype(np.int32)
    #qzs=((pzs-TOP_Z_MIN)//TOP_Z_DIVISION).astype(np.int32)
    qzs=(pzs-TOP_Z_MIN)/TOP_Z_DIVISION
    quantized = np.dstack((qxs,qys,qzs,prs)).squeeze()

    X0, Xn = 0, int((TOP_X_MAX-TOP_X_MIN)//TOP_X_DIVISION)+1
    Y0, Yn = 0, int((TOP_Y_MAX-TOP_Y_MIN)//TOP_Y_DIVISION)+1
    Z0, Zn = 0, int((TOP_Z_MAX-TOP_Z_MIN)/TOP_Z_DIVISION)
    height  = Xn - X0
    width   = Yn - Y0
    channel = Zn - Z0  + 2
    # print('height,width,channel=%d,%d,%d'%(height,width,channel))
    top = np.zeros(shape=(height,width,channel), dtype=np.float32)


    # histogram = Bin(channel, 0, Zn, "z", Bin(height, 0, Yn, "y", Bin(width, 0, Xn, "x", Maximize("intensity"))))
    # histogram.fill.numpy({"x": qxs, "y": qys, "z": qzs, "intensity": prs})

    if 1:  #new method
        for x in range(Xn):
            ix  = np.where(quantized[:,0]==x)
            quantized_x = quantized[ix]
            if len(quantized_x) == 0 : continue
            yy = -x

            for y in range(Yn):
                iy  = np.where(quantized_x[:,1]==y)
                quantized_xy = quantized_x[iy]
                count = len(quantized_xy)
                if  count==0 : continue
                xx = -y

                top[yy,xx,Zn+1] = min(1, np.log(count+1)/math.log(32))
                max_height_point = np.argmax(quantized_xy[:,2])
                top[yy,xx,Zn]=quantized_xy[max_height_point, 3]

                for z in range(Zn):
                    iz = np.where ((quantized_xy[:,2]>=z) & (quantized_xy[:,2]<=z+1))
                    quantized_xyz = quantized_xy[iz]
                    if len(quantized_xyz) == 0 : continue
                    zz = z

                    #height per slice
                    max_height = max(0,np.max(quantized_xyz[:,2])-z)
                    top[yy,xx,zz]=max_height
    return top

def lidar_to_top_fast(lidar):

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

    if (cfg.DATA_SETS_TYPE == 'didi' or cfg.DATA_SETS_TYPE == 'test' or cfg.DATA_SETS_TYPE=='didi2'):
        lidar=filter_center_car(lidar)

    # get index in grid
    lidar[:, 0] = ((lidar[:, 0]-TOP_X_MIN)//TOP_X_DIVISION).astype(np.int32)
    lidar[:, 1] = ((lidar[:, 1]-TOP_Y_MIN)//TOP_Y_DIVISION).astype(np.int32)
    lidar[:, 2] = ((lidar[:, 2]-TOP_Z_MIN)/TOP_Z_DIVISION).astype(np.float32)

    # make grid
    X0, Xn = 0, int((TOP_X_MAX-TOP_X_MIN)//TOP_X_DIVISION)+1
    Y0, Yn = 0, int((TOP_Y_MAX-TOP_Y_MIN)//TOP_Y_DIVISION)+1
    Z0, Zn = 0, int((TOP_Z_MAX-TOP_Z_MIN)/TOP_Z_DIVISION)
    height  = Xn - X0
    width   = Yn - Y0
    channel = Zn - Z0 + 2 # height map, intensity and density
    # make cell
    grid = np.zeros((height, width, 1), dtype=np.object)
    def _fill(x):
        if grid[int(x[0]), int(x[1]), 0] == 0:
            grid[int(x[0]), int(x[1]), 0] = []
        grid[int(x[0]), int(x[1]), 0].append(x)
    def _sort_by_height(x):
        x = x[0]
        if x != 0:
            h = np.array([i[2] for i in x])
            grid[int(x[0][0]), int(x[0][1]), 0] = list(np.array(x)[h.argsort(kind='heapsort')])
    np.apply_along_axis(_fill, 1, lidar)
    np.apply_along_axis(_sort_by_height, 2, grid)

    # make top
    top = np.zeros((height, width, channel), dtype=np.float32)
    def _cal_height(x):
        x = x[0]
        if x != 0:
            i = 0
            for z in range(Zn):
                if z+1 < x[i][2]: continue
                try:
                    while not (x[i][2] <= z+1 and x[i+1][2] > z+1):
                        i += 1
                except: # all the point are in this interval
                    top[int(x[0][0]), int(x[0][1]), z] = max(x[-1][2]-z, 0)
                    break

                top[int(x[0][0]), int(x[0][1]), z] = max(x[i][2]-z, 0)
                i += 1    
    def _cal_density(x):
        x = x[0]
        if x != 0:
            top[int(x[0][0]), int(x[0][1]), Zn+1] = min(1, np.log(len(x)+1)/np.log(32))
    def _cal_intensity(x): # there may be a moderate diff from origin version since if 2 point has the same height 
        x = x[0]
        if x != 0:
            top[int(x[0][0]), int(x[0][1]), Zn] = x[-1][3]
    np.apply_along_axis(_cal_height, 2, grid)
    np.apply_along_axis(_cal_density, 2, grid)    
    np.apply_along_axis(_cal_intensity, 2, grid)

    return top

def test_lidar_fast():
  #raw = np.fromfile('/data/mxj/kitti/object/training/velodyne/000000.bin', dtype=np.float32).reshape((-1, 4))
  raw = np.fromfile('/home/maxiaojian/data/kitti/2011_09_26/2011_09_26_drive_0001_sync/velodyne_points/data/0000000000.bin', dtype=np.float32).reshape((-1, 4))
  t = time.time()
  lidar_to_top_fast(raw)
  t = time.time() - t
  print('Costs:{}s'.format(t))
  t = time.time()
  lidar_to_front_fast(raw)
  t = time.time() - t
  print('Costs:{}s'.format(t))

def test_lidar():
  # raw = np.fromfile('/data/mxj/kitti/object/training/velodyne/000000.bin', dtype=np.float32).reshape((-1, 4))
  raw = np.fromfile('/home/maxiaojian/data/kitti/2011_09_26/2011_09_26_drive_0001_sync/velodyne_points/data/0000000000.bin', dtype=np.float32).reshape((-1, 4))
  t = time.time() 
  lidar_to_top(raw)
  t = time.time() - t
  print('Costs:{}s'.format(t))
  t = time.time()
  lidar_to_front(raw)
  t = time.time() - t
  print('Costs:{}s'.format(t))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='testing')

    parser.add_argument('-n', '--tag', type=str, nargs='?', default='unknown_tag',
                        help='set log tag')
    parser.add_argument('-t', '--target-dir', type=str, nargs='?', default='test_dir')
    parser.add_argument('-s', '--save-name', type=str, nargs='?', default='test')
    parser.add_argument('-i', '--interactive', action="store_true")


    args = parser.parse_args()

    print('\n\n{}\n\n'.format(args))

    # test_3dop()
    # test_rpn(args)
    test_rpn_origin_dataset(args)
    # test_lidar_fast()
    # test_lidar()
    # test_mv3d(args)
    # test_single_mv3d(args)
    # if args.interactive:
    #     test_rpn_target_interact(args)
    # else:
    #     test_rpn_target(args)
