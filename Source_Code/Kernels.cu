#include <cassert>
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cfloat>


#include "Kernels.h" // Include the header file for function declarations
#include "load_npy.h"

using namespace std;

// The mask and bias are declared in the header file

// CUDA kernel for 2D convolution
__global__ void Convolution_2D(float *d_input, float *d_output, const layer_DIM L) {

  int outRow = blockIdx.y * blockDim.y + threadIdx.y;
  int outCol = blockIdx.x * blockDim.x + threadIdx.x;
  int outDep = blockIdx.z * blockDim.z + threadIdx.z;

  if (outRow < L.output.x && outCol < L.output.y && outDep < L.output.z) {
      float sum = 0.0;

      // Perform convolution
      for (int maskRow = 0; maskRow < L.mask.x; ++maskRow) {
          for (int maskCol = 0; maskCol < L.mask.y; ++maskCol) {
              for (int maskDep = 0; maskDep < L.input.z; ++maskDep) {

                  int inputRow = outRow + maskRow;
                  int inputCol = outCol + maskCol;
                  int inputDep = maskDep;
                
                  if (inputRow < L.input.x && inputCol < L.input.y && inputDep < L.input.z) {

                      int inputIndex = (inputRow * L.input.y * L.input.z) + (inputCol * L.input.z) + inputDep;
                      int maskIndex = (outDep * L.mask.x * L.mask.y * L.mask.z) + (maskRow * L.mask.y * L.mask.z) + (maskCol * L.mask.z) + maskDep;


                      // int maskIndex = (outDep * L.mask.x * L.mask.y * L.mask.z) + (maskDep * L.mask.y * L.mask.x) + (maskRow * L.mask.y) + maskCol ;

                      sum += d_input[inputIndex] * mask[maskIndex];
                  }
              }
          }
      }

      // Add bias
      sum += bias[outDep];

      // Apply ReLU activation
      if (sum < 0.0f) {
          sum = 0.0f;
      }                   
      
      int outputIndex = outRow * (L.output.y * L.output.z) + outCol * L.output.z + outDep;
      d_output[outputIndex] = sum;
  }
}



void perform_convolution(layer_DIM &L_DIM, layer_data &L_DATA) {
    int size_input = L_DIM.input.x * L_DIM.input.y * L_DIM.input.z;
    int size_mask = L_DIM.mask.x * L_DIM.mask.y * L_DIM.mask.z * L_DIM.mask.num_filters;
    int size_bias = L_DIM.mask.num_filters;
    int size_output = L_DIM.output.x * L_DIM.output.y * L_DIM.output.z;

    int size_input_bytes = size_input * sizeof(float);
    int size_mask_bytes = size_mask * sizeof(float);
    int size_bias_bytes = size_bias * sizeof(float);
    int size_output_bytes = size_output * sizeof(float);
    
    if (sizeof(mask) < size_mask_bytes) {
        cerr << "\n\n\n\n\n\n\n\nInappropriate size of const mask. Exiting...\n Must be greater than :" << size_mask << "\n\n\n\n\n\n\n" << endl;
        exit(EXIT_FAILURE);
    }

    if (sizeof(bias) < size_bias_bytes) {
        cerr << "Inappropriate size of const bias. Exiting..." << endl;
        exit(EXIT_FAILURE);
    }

    // Device pointers
    float *d_input, *d_output;

    // Allocate device memory
    cudaMalloc(&d_input, size_input_bytes);
    cudaMalloc(&d_output, size_output_bytes);

    // Copy data from host to device
    cudaMemcpy(d_input, L_DATA.h_input, size_input_bytes, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(mask, L_DATA.h_mask, size_mask_bytes);
    cudaMemcpyToSymbol(bias, L_DATA.h_bias, size_bias_bytes);

    // Define block size and grid size
    int Threads = 8;
    dim3 blockSize(Threads, Threads, Threads);
    dim3 gridSize((L_DIM.output.x + blockSize.x - 1) / blockSize.x,
                  (L_DIM.output.y + blockSize.y - 1) / blockSize.y,
                  (L_DIM.output.z + blockSize.z - 1) / blockSize.z); // We launch one grid for each filter

    
    // Print block size and grid size
    cout << "Block Size: (" << blockSize.x << ", " << blockSize.y << ", " << blockSize.z << ")" << endl;
    cout << "Grid Size: (" << gridSize.x << ", " << gridSize.y << ", " << gridSize.z << ")" << endl;
    
    
    // Launch the convolution kernel with layer_DIM object
    Convolution_2D<<<gridSize, blockSize>>>(d_input, d_output, L_DIM);

    // Copy data from device to host
    cudaMemcpy(L_DATA.h_output, d_output, size_output_bytes, cudaMemcpyDeviceToHost);


    // Verify the output  
    // cout << "\n=================  Verifying Layer:" << L_DATA.index << " Results  =====================" << endl ;
    // verify_kernel_output(L_DATA, L_DIM);

    cout << "\n======================GPU_Result============================" << endl ;
    print_output(L_DATA.h_output, L_DIM, 1);

    // cout << "\n===================Tensor_Flow_Result=========================" << endl ;
    // print_output(L_DATA.v_output, L_DIM, 1);


    
    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    // Clean up host memory
    // delete[] L_DATA.h_input;
    // delete[] L_DATA.h_mask;
    // delete[] L_DATA.h_output;
}



// CUDA kernel for Max Pooling
__global__ void MaxPooling_2D(float* d_input, float* d_output, layer_DIM L, dim3 poolSize, dim3 stride) {
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int outDep = blockIdx.z * blockDim.z + threadIdx.z;

    // Calculate the starting position for pooling
    int inputRowStart = outRow * stride.x;
    int inputColStart = outCol * stride.y;

    if (outRow < L.output.x && outCol < L.output.y && outDep < L.output.z) {
        float maxVal = -FLT_MAX; // Initialize to negative infinity

        // Perform Max Pooling
        for (int row = 0; row < poolSize.x; ++row) {
            for (int col = 0; col < poolSize.y; ++col) {
                int inputRow = inputRowStart + row;
                int inputCol = inputColStart + col;

                if (inputRow < L.input.x && inputCol < L.input.y) {
                    int inputIndex = (inputRow * L.input.y * L.input.z) + (inputCol * L.input.z) + outDep;
                    maxVal = fmaxf(maxVal, d_input[inputIndex]);
                }
            }
        }

        // Store the result in the output
        int outputIndex = outRow * (L.output.y * L.output.z) + outCol * L.output.z + outDep;
        d_output[outputIndex] = maxVal;
    }
}


void perform_max_pooling(layer_DIM &L_DIM, layer_data &L_DATA, dim3 poolSize, dim3 stride) {
    // Allocate device memory
    float *d_input, *d_output;
    int input_size = L_DIM.input.x * L_DIM.input.y * L_DIM.input.z;
    int output_size = L_DIM.output.x * L_DIM.output.y * L_DIM.output.z;
    cudaMalloc((void**)&d_input, input_size * sizeof(float));
    cudaMalloc((void**)&d_output, output_size * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_input, L_DATA.h_input, input_size * sizeof(float), cudaMemcpyHostToDevice);

    // Kernel launch parameters
    dim3 blockDim(16, 16, 1);
    dim3 gridDim((L_DIM.output.x + blockDim.x - 1) / blockDim.x, (L_DIM.output.y + blockDim.y - 1) / blockDim.y, L_DIM.output.z);

    // Launch kernel
    MaxPooling_2D<<<gridDim, blockDim>>>(d_input, d_output, L_DIM, poolSize, stride);

    // Copy results back to host
    cudaMemcpy(L_DATA.h_output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);


    // Verify the output
    cout << "\n=================  Verifying Layer:" << L_DATA.index << " Results  =====================" << endl;
    verify_kernel_output(L_DATA, L_DIM); // Assuming you have a verification function
    
    cout << "\n======================GPU_Result============================" << endl ;
    print_output(L_DATA.h_output, L_DIM, 1);

    cout << "\n===================Tensor_Flow_Result=========================" << endl ;
    print_output(L_DATA.v_output, L_DIM, 1);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}




// __global__ void Dense(float *d_input, float *d_output, const layer_DIM L) {
//     int outIndex = threadIdx.x;

//     // Shared memory to store the sums for each output index
//     float shared_sums[10];

//     if (outIndex < L.output.x) {
//         float sum = 0.0f;

//         // Compute the linear combination of inputs and weights
//         for (int i = 0; i < L.input.x; ++i) {
//             int weightIndex = i * L.output.x + outIndex;
//             sum += d_input[i] * mask[weightIndex];
//         }

//         // Add bias
//         sum += bias[outIndex];

//         // Store the sum in shared memory
//         shared_sums[outIndex] = sum;

//         __syncthreads(); // Ensure all threads have written their sums

//         // Now, we will compute the softmax
//         float temp_sum = 0.0f;
//         float temp_max = -FLT_MAX;

//         // Determine the maximum value in shared_sums for numerical stability
//         for (int i = 0; i < L.output.x; ++i) {
//             temp_max = max(temp_max, shared_sums[i]);
//         }

//         __syncthreads(); // Ensure all threads have the max value

//         // Compute the exponentials and the sum of exponentials
//         for (int i = 0; i < L.output.x; ++i) {
//             temp_sum += exp(shared_sums[i] - temp_max);
//         }

//         __syncthreads(); // Ensure all threads have the sum of exponentials

//         // Compute the softmax value for this thread's output index
//         d_output[outIndex] = exp(shared_sums[outIndex] - temp_max) / temp_sum;
//     }
// }





__global__ void Dense(float *d_input, float *d_output, const layer_DIM L) {
    int outIndex = blockIdx.x * blockDim.x + threadIdx.x;

    // Shared memory to store logits for all threads in the block
    extern __shared__ float logits[];

    if (outIndex < L.output.x) {
        float sum = 0.0f;

        // Compute the linear combination of inputs and weights
        for (int i = 0; i < L.input.x; ++i) {
            int weightIndex = i * L.output.x + outIndex;
            sum += d_input[i] * mask[weightIndex];
        }

        // Add bias
        sum += bias[outIndex];

        // Store the logits in shared memory
        logits[outIndex] = sum;

        __syncthreads(); // Ensure all threads have written their logits

        // Compute the maximum value of logits for numerical stability
        float maxLogit = -FLT_MAX;
        for (int i = 0; i < L.output.x; ++i) {
            maxLogit = max(maxLogit, logits[i]);
        }

        __syncthreads(); // Ensure all threads have the maxLogit value

        // Subtract the maxLogit from each logit and compute the exponentials
        float expSum = 0.0f;
        for (int i = 0; i < L.output.x; ++i) {
            logits[i] = exp(logits[i] - maxLogit);
            expSum += logits[i];
        }

        __syncthreads(); // Ensure all threads have the expSum value

        // Divide by the sum of the exponentials to get the softmax probabilities
        d_output[outIndex] = logits[outIndex] / expSum;
    }
}





void perform_Dense(layer_DIM &L_DIM, layer_data &L_DATA) {
    int size_input = L_DIM.input.x;
    int size_output = L_DIM.output.x;
    int size_weights = L_DIM.mask.x * L_DIM.mask.y;
    int size_bias = L_DIM.bias;

    int size_input_bytes = size_input * sizeof(float);
    int size_output_bytes = size_output * sizeof(float);
    int size_weights_bytes = size_weights * sizeof(float);
    int size_bias_bytes = size_bias * sizeof(float);

    if (sizeof(mask) < size_weights_bytes) {
        cerr << "\n\n\n\n\n\n\n\nInappropriate size of const mask. Exiting...\n Must be greater than :" << size_weights << "\n\n\n\n\n\n\n" << endl;
        exit(EXIT_FAILURE);
    }

    if (sizeof(bias) < size_bias_bytes) {
        cerr << "Inappropriate size of const bias. Exiting..." << endl;
        exit(EXIT_FAILURE);
    }

    // Device pointers
    float *d_input, *d_output;

    // Allocate device memory
    cudaMalloc(&d_input, size_input_bytes);
    cudaMalloc(&d_output, size_output_bytes);

    // Copy data from host to device
    cudaMemcpy(d_input, L_DATA.h_input, size_input_bytes, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(mask, L_DATA.h_mask, size_weights_bytes);
    cudaMemcpyToSymbol(bias, L_DATA.h_bias, size_bias_bytes);

    // Define block size and grid size
    int Threads = 10; // Adjust based on your needs
    dim3 blockSize(Threads,1 ,1);
    dim3 gridSize((size_output + Threads - 1) / Threads, 1, 1);

    // Print block size and grid size
    cout << "Block Size: (" << blockSize.x << "x" << blockSize.y << "x" << blockSize.z << ")" << endl;
    cout << "Grid Size: (" << gridSize.x << "x" << gridSize.y  << "x" << gridSize.z << ")" << endl;

    // Launch the dense kernel
    Dense<<<gridSize, blockSize>>>(d_input, d_output, L_DIM);

    // Copy data from device to host
    cudaMemcpy(L_DATA.h_output, d_output, size_output_bytes, cudaMemcpyDeviceToHost);


    cout << fixed << setprecision(10); // Set precision to 5 decimal places
    // Verify the output
    cout << "\n=================  Verifying Layer:" << L_DATA.index << " Results  =====================" << endl;
    verify_kernel_output(L_DATA, L_DIM);


    cout << "\n======================GPU_Result============================" << endl;
    for(int a = 0 ; a < L_DIM.output.x ;  a ++ ){
      cout << L_DATA.h_output[a] << " " ;
    }
    cout << endl ;

    cout << "\n===================Tensor_Flow_Result=========================" << endl;
    for(int a = 0 ; a < L_DIM.output.x ;  a ++ ){
      cout << L_DATA.v_output[a] << " " ;
    }
    cout << endl ;

    // Find and print the index of the maximum value in L_DATA.h_output
    int maxIndex = 0;
    for (int i = 1; i < L_DIM.output.x; i++) {
        if (L_DATA.h_output[i] > L_DATA.h_output[maxIndex]) {
            maxIndex = i;
        }
    }

    cout << "------------------------------------------------------" << endl;
    cout << "Index of maximum value in GPU_Result: " << maxIndex << endl;
    cout << "------------------------------------------------------" << endl;


    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}







void perform_flatting(layer_DIM &L_DIM, layer_data &L_DATA, layer_DIM &L_Input_DIM) {
    
    // Traverse through each depth, row, and column to flatten the input data
    for (int i = 0; i < L_Input_DIM.output.z; i++) {  // depth
        for (int j = 0; j < L_Input_DIM.output.x; j++) {  // row
            for (int k = 0; k < L_Input_DIM.output.y; k++) {  // column
                
                int input_index =  (j * L_Input_DIM.output.y * L_Input_DIM.output.z) + (k * L_Input_DIM.output.z) + i;
                // int output_index = (i * L_Input_DIM.output.x * L_Input_DIM.output.y) + (j * L_Input_DIM.output.y) + k;

                L_DATA.h_output[input_index] = L_DATA.h_input[input_index];
            }
        }
    }

    // Verify the output
    cout << "\n=================  Verifying Layer:" << L_DATA.index << " Results  =====================" << endl;
    verify_kernel_output(L_DATA, L_DIM); // Assuming you have a verification function
    
    cout << "\n======================Input============================" << endl;
    for(int a = 0 ; a < L_DIM.output.x ;  a ++ ){
      cout << L_DATA.h_input[a] << " " ;
    }
    cout << endl ;
    
    cout << "\n======================GPU_Result============================" << endl;
    for(int a = 0 ; a < L_DIM.output.x ;  a ++ ){
      cout << L_DATA.h_output[a] << " " ;
    }
    cout << endl ;

    cout << "\n===================Tensor_Flow_Result=========================" << endl;
    for(int a = 0 ; a < L_DIM.output.x ;  a ++ ){
      cout << L_DATA.v_output[a] << " " ;
    }
    cout << endl ;
}






