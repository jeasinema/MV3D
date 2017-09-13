#include "cuda.h"

// for old version titan x, the same as 1080
// pycuda._driver.device_attribute.MAX_THREADS_PER_BLOCK: 1024
// pycuda._driver.device_attribute.MAX_BLOCK_DIM_X: 1024
// pycuda._driver.device_attribute.MAX_BLOCK_DIM_Y: 1024
// pycuda._driver.device_attribute.MAX_BLOCK_DIM_Z: 64
// pycuda._driver.device_attribute.MAX_GRID_DIM_X: 2147483647
// pycuda._driver.device_attribute.MAX_GRID_DIM_Y: 65535
// pycuda._driver.device_attribute.MAX_GRID_DIM_Z: 65535

__device__ float integration(float *data, int length, int channel_amount)
{

}

__global__ void remove_empty(float *inds, int *anchors, float *view, int *anchors_shape, int *view_shape)
{
    
}
