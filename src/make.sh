cd ./net/lib/
python3 ./setup.py build_ext --inplace
./make.sh
cd ../../

ln -nfs $(pwd)/net/lib/roi_pooling_layer/roi_pooling.so ./net/roipooling_op/roi_pooling.so
ln -nfs $(pwd)/net/lib/nms/gpu_nms.cpython-35m-x86_64-linux-gnu.so ./net/processing/gpu_nms.cpython-35m-x86_64-linux-gnu.so
ln -nfs $(pwd)/net/lib/nms/cpu_nms.cpython-35m-x86_64-linux-gnu.so ./net/processing/cpu_nms.cpython-35m-x86_64-linux-gnu.so
ln -nfs $(pwd)/net/lib/utils/bbox.cpython-35m-x86_64-linux-gnu.so ./net/processing/cython_bbox.cpython-35m-x86_64-linux-gnu.so

# compile remove empty box kernel
GPU_ARCH="sm_61"
nvcc --cubin -arch $GPU_ARCH -I /usr/local/lib/python3.5/dist-packages/pycuda-2017.1.1-py3.5-linux-x86_64.egg/pycuda/cuda ./net/utility/remove_empty_box_kernel.cu -o ./net/utility/remove_empty_box_kernel.cubin
nvcc --cubin -arch $GPU_ARCH -I /usr/local/lib/python3.5/dist-packages/pycuda-2017.1.1-py3.5-linux-x86_64.egg/pycuda/cuda ./net/utility/front_top_kernel.cu -o ./net/utility/front_top_kernel.cubin
