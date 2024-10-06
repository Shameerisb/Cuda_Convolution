#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <iomanip>
#include "load_npy.h" // Include the necessary headers

using namespace std;

// Function declarations
void verify_output(const layer_data& L_data, const layer_DIM& L);
void verify_kernel_output(const layer_data& L_data, const layer_DIM& L);
void print_test(const layer_DIM &First_layer_DIM, const layer_data& First_layer_data);
void print_output(const float* output, const layer_DIM& L, int maps);



#endif // UTILS_H
