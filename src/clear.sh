rm ./net/roipooling_op/roi_pooling.so
rm ./net/lib/psroi_pooling_layer/psroi_pooling.so
rm ./net/processing/gpu_nms.cpython-35m-x86_64-linux-gnu.so
rm ./net/processing/cpu_nms.cpython-35m-x86_64-linux-gnu.so
rm ./net/lib/pycocotools/_mask.cpython-35m-x86_64-linux-gnu.so
rm ./net/processing/cython_bbox.cpython-35m-x86_64-linux-gnu.so
rm ./net/lib/nms/*.so
rm ./net/lib/roi_pooling_layer/roi_pooling.so
rm ./net/lib/utils/*.so

#.c
rm ./net/lib/nms/cpu_nms.c
rm ./net/lib/utils/bbox.c

#.cpp
rm ./net/lib/nms/gpu_nms.cpp

rm -Rf ./net/lib/build
