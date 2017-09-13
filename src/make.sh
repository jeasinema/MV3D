cd ./net/lib/
python ./setup.py build_ext --inplace
./make.sh
cd ../../

ln -s $(pwd)/net/lib/roi_pooling_layer/roi_pooling.so ./net/roipooling_op/roi_pooling.so
ln -s $(pwd)/net/lib/nms/gpu_nms.cpython-36m-x86_64-linux-gnu.so ./net/processing/gpu_nms.cpython-36m-x86_64-linux-gnu.so
ln -s $(pwd)/net/lib/nms/cpu_nms.cpython-36m-x86_64-linux-gnu.so ./net/processing/cpu_nms.cpython-36m-x86_64-linux-gnu.so
ln -s $(pwd)/net/lib/utils/bbox.cpython-36m-x86_64-linux-gnu.so ./net/processing/cython_bbox.cpython-36m-x86_64-linux-gnu.so

# compile remove empty box kernel
GPU_ARCH="sm_61"
nvcc --cubin -arch $GPU_ARCH -I /home/maxiaojian/anaconda3/lib/python3.6/site-packages/pycuda-2017.1.1-py3.6-linux-x86_64.egg/pycuda/cuda ./net/utility/remove_empty_box_kernel.cu -o ./net/utility/remove_empty_box_kernel.cubin
