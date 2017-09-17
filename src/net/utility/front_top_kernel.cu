#include "cuda.h"

__global__ void lidar_to_top(float *top, int *top_shape, int *lidar_x, 
    int *lidar_y, float *lidar_z, float *lidar_i, int *lidar_shape)
{
    // input:
    // sorted lidar_x/y/z/i (by the order of x, y, z)
    // x,y can be used as index in top output 
    int height = top_shape[0];
    int width = top_shape[1];
    int channel = top_shape[2];
    int point_amount = lidar_shape[0];

    int c = threadIdx.x; // the chann that cur thread responsible for. 
    int height_lb = c; // lb <= z <= hb
    int height_hb = c+1;
    int index = blockIdx.x;

    int x = height - lidar_x[index] - 1;
    int y = width - lidar_y[index] - 1;
    float z = lidar_z[index];
    float i = lidar_i[index];

    int x_n = -1;
    int y_n = -1;
    float z_n = -1; 
   
    if (index < point_amount-1) {
        x_n = height - lidar_x[index + 1] - 1;
        y_n = width -lidar_y[index + 1] - 1;
        z_n = lidar_z[index + 1];
    }

    if (x < 0 || x >= height) return;
    if (y < 0 || y >= width) return;
    // intensity and density channel do not cal seperately
    if (c < 0 || c >= channel-2) return;
    
    // fill in height channel 
    int base_index = (x*width + y)*channel;
    int top_index = base_index + c;
    int top_intensity_index = base_index + channel - 2;

    if (z < height_lb || z > height_hb) return; // not belong to cur channel 
    if (x_n == x && y_n == y && height_lb <= z_n && z_n <= height_hb) return; // turn to next point 
    top[top_index] = z - height_lb; 
    // fill in intensity channel 
    if (x_n != x || y_n != y) {  // this is the last one in cur grid 
        top[top_intensity_index] = i;
    }
}

__global__ void lidar_to_top_density(float *top_density, int *lidar_x, int *lidar_y, int *lidar_shape, int *top_shape)
{
    int point_amount = lidar_shape[0];
    int height = top_shape[0];
    int width = top_shape[1];
    for (int i = 0; i < point_amount; ++i) {
        int x = height - lidar_x[i] - 1;
        int y = width - lidar_y[i] - 1;
        int base_index = x*width + y;
        top_density[base_index] ++;
    }
}

__global__ void lidar_to_front(float *front, int *front_shape, float *lidar, int *lidar_shape)
{
}
