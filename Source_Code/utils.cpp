#include <cmath>
#include "utils.h"

// // Verification function
// void verify_output(const layer_data& L_data, const layer_DIM& L) {
//     const float tolerance = 1e-6f; // Tolerance for floating-point comparison
//     bool verification_passed = true; // Flag to track verification status

//     cout << "\nCPU : Verifying output..." << endl;

//     // Loop through each filter
//     for (int filter = 0; filter < L.mask.num_filters; ++filter) {
//         // Loop through output dimensions
//         for (int row = 0; row < L.output.x; ++row) {
//             for (int col = 0; col < L.output.y; ++col) {
//                 float sum = 0.0f;

//                 // Perform convolution manually for verification
//                 for (int k = 0; k < L.input.z; ++k) {
//                     for (int i = 0; i < L.mask.x; ++i) {
//                         for (int j = 0; j < L.mask.y; ++j) {
//                             int inputRow = row + i;
//                             int inputCol = col + j;
//                             // Ensure input indices are within bounds
//                             if (inputRow < L.input.x && inputCol < L.input.y) {
//                                 int inputIndex = (inputRow * L.input.y * L.input.z) + (inputCol * L.input.z) + k;
//                                 int maskIndex = (filter * L.mask.x * L.mask.y * L.mask.z) + (i * L.mask.y * L.mask.z) + (j * L.mask.z) + k;
//                                 sum += L_data.h_input[inputIndex] * L_data.h_mask[maskIndex];
//                             }
//                         }
//                     }
//                 }

//                 // Add bias
//                 sum += L_data.h_bias[filter];

//                 // Apply ReLU activation
//                 if (sum < 0.0f) {
//                     sum = 0.0f;
//                 }

//                 // Get the actual output from the GPU
//                 int outputIndex = (filter * L.output.x * L.output.y) + (row * L.output.y) + col;
//                 float actual_output = L_data.h_output[outputIndex];

//                 // Compare the expected output with the actual output
//                 if (fabs(actual_output - sum) >= tolerance) {
//                     verification_passed = false;
//                     // Commented out detailed printing
//                     // cout << "Verification failed at filter: " << filter << ", row: " << row << ", col: " << col << endl;
//                     // cout << "Expected output: " << sum << ", Actual output: " << actual_output << endl;
//                 }
//             }
//         }
//     }

//     if (verification_passed) {
//         cout << "Verification passed." << endl;
//     } else {
//         cout << "Verification failed." << endl;
//     }
// }





void verify_kernel_output(const layer_data& L_data, const layer_DIM& L) {
    const float tolerance = 1e-5f; // Tolerance for floating-point comparison
    bool verification_passed = true; // Flag to track verification status

    cout << "\nTensor_Flow : Verifying kernel output with v_output..." << endl;

    int size_output = L.output.x * L.output.y * L.output.z;

    for (int i = 0; i < size_output; ++i) {
      float difference = fabs(L_data.h_output[i] - L_data.v_output[i]);
        if (fabs(L_data.h_output[i] - L_data.v_output[i]) >= tolerance) {
            verification_passed = false;
            // // Commented out detailed printing
            // cout << "Mismatch at index " << i << endl;
            // cout << "Kernel output: " << L_data.h_output[i] << ", v_output: " << L_data.v_output[i] << endl;
            // cout << "Difference: " << difference << endl;
        }
    }

    if (verification_passed) {
        cout << "Verification with v_output passed." << endl;
    } else {
        cout << "Verification with v_output failed." << endl;
    }
    cout << "============================================================" << endl;
}

void print_output(const float* output, const layer_DIM& L, int maps) {
    cout << std::fixed << std::setprecision(3); // Set precision to 3 decimal places

    for (int filter = 0; filter < maps; ++filter) {
        cout << "Filter " << filter << " Results:" << endl;
        for (int row = 0; row < L.output.x; ++row) {
            for (int col = 0; col < L.output.y; ++col) {
                int index = row * (L.output.y * L.output.z) + col * L.output.z + filter;
                cout << output[index] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }
}

void print_test(const layer_DIM &L_DIM, const layer_data& L_DATA) {
    cout << "\n\n\n\n=====  Printing Testing Values For Structure Verification  =====" << endl;

    cout << "Input Dimensions: " << L_DIM.input.x << ", " << L_DIM.input.y << ", " << L_DIM.input.z << endl;
    cout << "Mask Dimensions: " << L_DIM.mask.x << ", " << L_DIM.mask.y << ", " << L_DIM.mask.z << ", " << L_DIM.mask.num_filters << endl;
    cout << "Output Dimensions: " << L_DIM.output.x << ", " << L_DIM.output.y << ", " << L_DIM.output.z << endl;

    int size_input = L_DIM.input.x * L_DIM.input.y * L_DIM.input.z;
    int size_mask = L_DIM.mask.x * L_DIM.mask.y * L_DIM.mask.z * L_DIM.mask.num_filters;
    int size_output = L_DIM.output.x * L_DIM.output.y * L_DIM.output.z;

    int size_input_bytes = size_input * sizeof(float);
    int size_mask_bytes = size_mask * sizeof(float);
    int size_output_bytes = size_output * sizeof(float);

    cout << "\n\nSize of input: " << size_input << endl;
    cout << "Size of mask: " << size_mask << endl;
    cout << "Size of output: " << size_output << endl;

    cout << "Size of input in bytes: " << size_input_bytes << endl;
    cout << "Size of mask in bytes: " << size_mask_bytes << endl;
    cout << "Size of output in bytes: " << size_output_bytes << endl;

    // Print all elements of weights (h_mask)
    cout << "\n\nWeights (h_mask): ";
    for (int i = 0; i < size_mask; ++i) {
        cout << L_DATA.h_mask[i] << " ";
    }
    cout << endl;

    // Print all elements of biases (h_bias)
    int size_bias = L_DIM.mask.num_filters; // Assuming bias has the same size as the number of filters
    cout << "Biases (h_bias): ";
    for (int i = 0; i < size_bias; ++i) {
        cout << L_DATA.h_bias[i] << " ";
    }

    // Print all elements of input in the format 28 x 28 x 1
    int width  = L_DIM.input.x;
    int height = L_DIM.input.y;
    int depth  = L_DIM.input.z;

    cout << "\n\nInput data (" << width <<  " x " << height <<  " x " <<  depth<< " ):" << endl;
    cout << fixed << setprecision(3); // Set precision to 5 decimal places

    for(int a = 0 ; a < 1 ; a++){
      cout << "\n\nDepth " << a << ":" << endl;
      for(int b = 0 ; b < height ; b ++) {
        for(int c = 0 ; c < width ; c++) {

          cout << L_DATA.h_input[(b * width * depth) + (c * depth) + a] << " "; 
        }
        cout << endl;
      }
      cout << endl;
    }

    cout<< "=========================================================== " << endl;
    cout<< "=========================================================== " << endl;
}
