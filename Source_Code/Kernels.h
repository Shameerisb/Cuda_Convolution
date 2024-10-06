#ifndef KERNELS_H
#define KERNELS_H

#include "utils.h"

__constant__ float mask[1024 * 10];
__constant__ float bias[1024 * 1];
__shared__ float logits[10];

void perform_convolution(layer_DIM &L_DIM, layer_data &L_DATA);
void perform_max_pooling(layer_DIM &L_DIM, layer_data &L_DATA, dim3 poolSize, dim3 stride);

void perform_flatting(layer_DIM &L_DIM, layer_data &L_DATA, layer_DIM &L_Input_DIM);
void perform_Dense(layer_DIM &L_DIM, layer_data &L_DATA);

__global__ void Convolution_2D(float *d_input, float *d_output, const layer_DIM L);

__global__ void MaxPooling_2D(float* d_input, float* d_output, layer_DIM L_DIM, dim3 poolSize, dim3 stride);

__global__ void Dense(float *d_input, float *d_output, const layer_DIM L);
#endif // KERNELS_H