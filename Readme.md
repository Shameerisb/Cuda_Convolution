# Project: CUDA and C++ Kernels for Image Processing

This project contains multiple C++ and CUDA kernel source files used for various computations. These source files are called by the `main.cu` file, which acts as the entry point for compiling and running the project. The entire project is compiled and run in the provided Colab notebook.

## Table of Contents

- [main.cu](#maincu) 
- [load_npy.cpp](#load_npycpp) 
- [utils.cpp](#utilscpp) 
- [Kernels.cu](#kernelscu) 
- [colab_notebook.ipynb](#colab_notebookipynb) 
---
---

# main.cu

This file serves as the entry point for executing a Convolutional Neural Network (CNN) using CUDA. It defines the structure and flow of operations performed through various layers, from convolution to dense layers.

## Includes

- Standard libraries for input/output and memory management.
- CUDA runtime and core libraries for GPU operations.
- Custom headers for kernel functions, utility functions, loading NumPy arrays, and CuDNN implementations.

## Main Function

### Initialization

- **Image Index**: An integer (`image_indextoload`) specifies which image to load for processing.
- **Layer Dimensions and Data Structures**: 
  - Six sets of `layer_DIM` and `layer_data` structures are defined for the different layers of the CNN.

### Layer Operations

1. **First Conv2D Layer**:
   - Calls `populate_input_layer` to initialize the input layer dimensions and data.
   - Loads the specified image using `Load_image`.
   - Performs convolution using `perform_convolution`.

2. **Second Conv2D Layer**:
   - Populates layer dimensions and data from the first layer.
   - Executes convolution on the second layer.

3. **Third Conv2D Layer**:
   - Similar to the second layer, it populates data from the second layer and performs convolution.

4. **Fourth Max Pooling Layer**:
   - Populates layer data from the third convolutional layer.
   - Defines pooling size and stride.
   - Calls `perform_max_pooling` to down-sample the feature maps.

5. **Fifth Flatten Layer**:
   - Populates layer data from the fourth layer (assumes the array is already flattened).
   - **Note**: The flattening step is commented out in the provided code.

6. **Sixth Dense Layer**:
   - Populates layer dimensions and data from the fourth layer.
   - Executes the dense layer operation using `perform_Dense`.

### Output

- The program concludes with a message indicating completion of processing.

## Notes

- The code is modular, allowing for easy integration of various layers and operations.
- Several commented-out sections suggest potential testing or alternative implementations (e.g., CuDNN convolution).
- The utility functions are expected to handle image loading, data verification, and output printing, though they are not detailed in this file.





---
---
# load_npy.cpp

The `load_npy.h` file is responsible for loading and managing data from `.npy` files. These files contain trained weights, input data, and other layer-specific information for a neural network. This header file includes several functions to manage data population and verification for different layers of the network. Here's a breakdown of the key functionalities:

## Functions

### 1. `Load_image`
- **Description**: Loads input image data from a `.npy` file and stores it in the `h_input` member of `layer_data`.
- **Functionality**:
  - Uses the `image_index` to select and copy the specific image's data into memory.

### 2. `populate_input_with_maps`
- **Description**: Copies output data from one layer (`L_DATA_input`) to the input data of the current layer (`L_DATA`).
- **Functionality**:
  - Based on the calculated input size, this function populates the input of the current layer with data from the previous layer.

### 3. `populate_verification`
- **Description**: Loads verification data from a `.npy` file into the `v_output` member of `layer_data`.
- **Functionality**:
  - Uses the `L_index` and `image_index` to retrieve specific layer verification data.

### 4. `populate_input_layer`
- **Description**: Populates input and output dimensions for a layer using `.npy` files containing the input and output shapes.
- **Functionality**:
  - Loads weights and biases from `.npy` files if they exist, and stores them in `h_mask` and `h_bias`, respectively.
  - If weights or biases are not found, the layer is initialized with default dimensions and values.
  - This function also handles layer-specific dimensions like convolution and dense layers.

### 5. `populate_layer`
- **Description**: A higher-level function that sets up a layer for processing.
- **Functionality**:
  - Calls both `populate_input_layer` and `populate_input_with_maps` to fully prepare a layer for use in the neural network.

---
---





# utils.cpp

The `utils.cpp` file contains utility functions for verifying the output of a neural network kernel, printing outputs, and displaying testing values for structure verification. Below is a breakdown of the key functionalities:

## Functions

### 1. `verify_kernel_output`
- **Description**: Verifies the output of the neural network kernel against the expected verification output (`v_output`).
- **Parameters**:
  - `const layer_data& L_data`: Contains the data structure for the layer, including the kernel output and verification output.
  - `const layer_DIM& L`: Dimensions of the layer, used for calculating the output size.
- **Functionality**:
  - Compares each element of the kernel output (`h_output`) with the corresponding verification output (`v_output`) using a defined tolerance.
  - Reports whether the verification passed or failed.

### 2. `print_output`
- **Description**: Prints the output results of the neural network layer.
- **Parameters**:
  - `const float* output`: Pointer to the output data to be printed.
  - `const layer_DIM& L`: Dimensions of the output layer.
  - `int maps`: Number of filters (maps) in the layer.
- **Functionality**:
  - Iterates through the output data and prints the results for each filter in a formatted manner.

### 3. `print_test`
- **Description**: Prints testing values for structure verification, including dimensions and sizes of input, mask (weights), and output data.
- **Parameters**:
  - `const layer_DIM& L_DIM`: Dimensions of the layer, including input, mask, and output.
  - `const layer_data& L_DATA`: Contains data for the layer, including weights (`h_mask`), biases (`h_bias`), and input data (`h_input`).
- **Functionality**:
  - Displays dimensions and sizes of input, mask, and output data, both in terms of counts and bytes.
  - Prints all elements of the weights and biases.
  - Displays input data in a specified format (e.g., 28 x 28 x 1).

## Key Features
- **Floating-Point Verification**: Uses a small tolerance for comparing floating-point numbers to account for numerical precision issues.
- **Formatted Output**: Sets precision for floating-point values to ensure clarity when printing results.
- **Depth-wise Input Visualization**: Prints the input data depth-wise for easier visualization of the 3D structure.

### Example Output
- The functions provide outputs such as verification status, sizes of various components, and the contents of weights, biases, and input data.


---
---



# Kernels.cu

This file contains CUDA kernels and functions for performing various operations in a neural network, including convolution, max pooling, and dense layer processing. Below is a breakdown of the key components:

## Includes

- Standard and CUDA libraries for error handling, memory management, and mathematical operations.

## Kernel Functions

### Convolution_2D

- **Purpose**: Executes 2D convolution on input data.
- **Parameters**:
  - `float *d_input`: Device pointer to the input data.
  - `float *d_output`: Device pointer to the output data.
  - `const layer_DIM L`: Structure containing layer dimensions.
- **Details**:
  - Loops through output dimensions and applies the convolution operation using a defined mask and bias.
  - Applies ReLU activation after computing the convolution.

### MaxPooling_2D

- **Purpose**: Performs max pooling operation on input data.
- **Parameters**:
  - `float *d_input`: Device pointer to the input data.
  - `float *d_output`: Device pointer to the output data.
  - `layer_DIM L`: Structure containing layer dimensions.
  - `dim3 poolSize`: Dimensions of the pooling window.
  - `dim3 stride`: Stride size for the pooling operation.
- **Details**:
  - Iterates through the input data to determine the maximum value within the specified pooling window and stores it in the output.

### Dense

- **Purpose**: Implements the fully connected layer's forward pass.
- **Parameters**:
  - `float *d_input`: Device pointer to the input data.
  - `float *d_output`: Device pointer to the output data.
  - `const layer_DIM L`: Structure containing layer dimensions.
- **Details**:
  - Computes the linear combination of inputs and weights, adds bias, and then applies the softmax function to produce probabilities.

## Host Functions

### perform_convolution

- **Purpose**: Manages memory allocation, kernel invocation, and data transfer for the convolution operation.
- **Parameters**:
  - `layer_DIM &L_DIM`: Reference to layer dimensions.
  - `layer_data &L_DATA`: Reference to layer data, including inputs, outputs, and parameters.
- **Details**:
  - Allocates device memory, copies data from host to device, launches the convolution kernel, and retrieves results.

### perform_max_pooling

- **Purpose**: Manages memory allocation and kernel invocation for the max pooling operation.
- **Parameters**:
  - `layer_DIM &L_DIM`: Reference to layer dimensions.
  - `layer_data &L_DATA`: Reference to layer data.
  - `dim3 poolSize`: Pooling dimensions.
  - `dim3 stride`: Stride size for pooling.
- **Details**:
  - Similar to `perform_convolution`, but specifically for max pooling.

### perform_Dense

- **Purpose**: Manages memory allocation and kernel invocation for the dense layer.
- **Parameters**:
  - `layer_DIM &L_DIM`: Reference to layer dimensions.
  - `layer_data &L_DATA`: Reference to layer data.
- **Details**:
  - Allocates device memory, copies data from host to device, launches the dense kernel, and retrieves results.

## Notes

- Memory management is critical; device memory is allocated and freed appropriately.
- Error checks are performed for mask and bias sizes to ensure they are sufficient.
- The code employs shared memory for the Dense kernel to improve performance during the softmax computation.
































# colab_notebook.ipynb
This notebook provides the environment setup and compilation instructions for running the project. It includes:
- Instructions to set up CUDA on Colab.
- Commands for compiling the source files (`main.cu`, `file1.cu`, `file2.cu`, `file3.cu`).
- Example usage and output of the program.

Make sure to follow the steps provided in the notebook for running the project in a Colab environment.

---

## How to Run
1. Clone the repository to your local machine or open it in Google Colab.
2. Open the `colab_notebook.ipynb` and follow the instructions for setting up the environment.
3. Run all the cells to compile and execute the CUDA kernels.
