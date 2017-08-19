import mv3d
import mv3d_net
import glob
from config import *
# import utils.batch_loading as ub
import argparse
import os
from utils.training_validation_data_splitter import TrainingValDataSplitter
from utils.batch_loading import Loading3DOP


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='testing')

    parser.add_argument('-n', '--tag', type=str, nargs='?', default='unknown_tag',
                        help='set log tag')

    args = parser.parse_args()

    print('\n\n{}\n\n'.format(args))
  
    # 3dop_proposal should be (N, 8) (7 for pos, 1 for objectness score)
    with Loading3DOP(object_dir='/data/mxj/kitti/object_3dop', proposals_dir='/data/mxj/kitti/3dop_proposal', queue_size=20, require_shuffle=False, 
         is_testset=True) as testset:
      test = mv3d.Tester_3DOP(*testset.get_shape(), log_tag=args.tag)
      data = testset.load()
      count = 0
      while data:
        boxes, labels = test(*data)
        data = testset.load()
        print("Process {} data".format(count))
        boxes, labels = np.array(boxes), np.array(labels)
        np.save(os.path.join('./tmp', str(count) + "_boxes.npy"), boxes)
        np.save(os.path.join('./tmp', str(count) + "_labels.npy"), labels)

        count += 1