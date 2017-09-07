#include "cuda.h"

__device__ float integration(float *data, int length, int channel_amount)
{
    float sum = 0;
    for (int i = 0; i < length; i++) {
        sum += data[i*channel_amount];
    }
    return sum;
}

__global__ void remove_empty(float *inds, int *anchors, float *view, int *anchors_shape, int *view_shape)
{
    int anchor_id = blockIdx.x;
    if (anchor_id >= anchors_shape[0]) return;
    int *anchor_base = anchors + 4*anchor_id;
    int x1 = anchor_base[0];
    int y1 = anchor_base[1];
    int x2 = anchor_base[2];
    int y2 = anchor_base[3];

    if (x1 < 0) x1 = 0;
    if (x1 >= view_shape[0]) x1 = view_shape[0]-1;
    if (x2 < 0) x2 = 0;
    if (x2 >= view_shape[0]) x2 = view_shape[0]-1;
    if (y1 < 0) y1 = 0;
    if (y1 >= view_shape[1]) y1 = view_shape[1]-1;
    if (y2 < 0) y2 = 0;
    if (y2 >= view_shape[1]) y2 = view_shape[1]-1;

    int anchor_w = x2 - x1;
    int anchor_l = y2 - y1;
    int channel = threadIdx.x;
    if (channel >= view_shape[2]) return;
    int line_id = blockIdx.y;
    if (line_id >= anchor_w) return;
    
    float *pos = view + ((x1 + line_id)*view_shape[1] + y1)*view_shape[2] + channel;
    int length = anchor_l;
    int channel_amount = view_shape[2];
    *(inds + channel_amount*anchor_id + channel) += integration(pos, length, channel_amount);
}
