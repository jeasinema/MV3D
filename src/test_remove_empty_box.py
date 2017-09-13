import argparse 
import mv3d
import mv3d_net
from utils.batch_loading import BatchLoading3
from config import *
import tensorflow as tf
from net.utility.remove_empty_box import remove_empty_anchor

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training')

    all= '%s,%s,%s,%s' % (mv3d_net.top_view_rpn_name ,mv3d_net.imfeature_net_name,mv3d_net.fusion_net_name, mv3d_net.frontfeature_net_name)

    parser.add_argument('-w', '--weights', type=str, nargs='?', default=all,  # FIXME
        help='use pre trained weights example: -w "%s" ' % (all))

    parser.add_argument('-t', '--targets', type=str, nargs='?', default=all,
        help='train targets example: -w "%s" ' % (all))

    parser.add_argument('-i', '--max_iter', type=int, nargs='?', default=1000,
                        help='max count of train iter')

    parser.add_argument('-n', '--tag', type=str, nargs='?', default='unknown_tag',
                        help='set log tag')

    parser.add_argument('-c', '--continue_train', action='store_true', default=False,
                        help='set continue train flag')

    parser.add_argument('-b', '--batch-size', type=int, nargs='?', default=1,
                        help='set continue train flag')

    parser.add_argument('-l', '--lr', type=float, nargs='?', default=0.001,
                        help='set learning rate')

    args = parser.parse_args()

    training_dataset = {
        '2011_09_26': [
          '0070', '0015', '0052', '0035', '0061', '0002', '0018', '0013', '0032', '0056', '0017', '0011',
          '0001', '0005', '0014', '0020', ' 0059',
          '0019', '0084', '0028', '0051', '0060', '0064', '0027', '0086', '0022', '0023', '0046', '0029', '0087', '0091'
        ]
    }

    validation_dataset = {
        '2011_09_26': [
          '0036', '0057', '0079', '0048', '0039', '0093'
        ]
    }
    tag = args.tag
    if tag == 'unknown_tag':
        tag = input('Enter log tag : ')
        print('\nSet log tag :"%s" ok !!\n' %tag)

    max_iter = args.max_iter
    weights=[]
    if args.weights != '':
        weights = args.weights.split(',')

    targets=[]
    if args.targets != '':
        targets = args.targets.split(',')


    import time
    import random
    import numpy as np
    with BatchLoading3(tags=training_dataset, require_shuffle=True, use_precal_view=True, queue_size=1, use_multi_process_num=0) as training:
      with BatchLoading3(tags=validation_dataset, require_shuffle=False, use_precal_view=True, queue_size=1, use_multi_process_num=0) as validation:
            print('loading graph')
            train = mv3d.Trainer(train_set=training, validation_set=validation,
                                 pre_trained_weights=weights, train_targets=targets, log_tag=tag,
                                 continue_train = args.continue_train, batch_size=args.batch_size, lr=args.lr)
            print('end loading graph')
            train(max_iter=2) 
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
   # view = np.load('/home/mxj/1.npy')[0]
   # anchors = np.load('/home/mxj/2.npy')
    view = 0.01*np.random.randn(800, 600, 27).astype(np.float32)
    t = time.time()
    index = remove_empty_anchor(view, anchors, 0)
    print('done, {}'.format(time.time()-t))
    print(index)
    print(len(index))
    train(max_iter=1000) 
