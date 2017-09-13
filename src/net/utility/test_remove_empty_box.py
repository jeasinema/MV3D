import tensorflow as tf
from remove_empty_box import remove_empty_anchor

if __name__ == '__main__':
    import time
    import random
    import numpy as np
    sess = tf.InteractiveSession()
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
    tf.ones(10).eval()
