# CUDA and C++ Kernels for Image Processing

This project involves implementing image processing using CUDA and C++ kernels, with the goal of executing operations like convolution, pooling, and dense layer computations. The source files are organized and compiled within a Colab notebook, which also serves as the execution environment.

---

## Table of Contents

- [Project Overview](#project-overview)
- [How to Run in Google Colab](#how-to-run-in-google-colab)
- [Files and Descriptions](#files-and-descriptions)
  - [main.cu](#maincu)
  - [load_npy.cpp](#load_npycpp)
  - [utils.cpp](#utilscpp)
  - [Kernels.cu](#kernelscu)
  - [colab_notebook.ipynb](#colab_notebookipynb)

---

## Project Overview

This project utilizes CUDA for GPU-accelerated image processing operations. The core functionality is implemented in C++ and CUDA kernels, which are compiled and executed through a Colab notebook. The project operates on a Convolutional Neural Network (CNN) structure, with several layers including convolution, max pooling, and fully connected layers.

---

## How to Run in Google Colab

- Open the Colab notebook [here](Copy_of_Mnist_in_CUDA.ipynb).
- Clone the repositories in the first code cell.
- Download the MNIST data and save files through the second code cell.
- Compile the code file and execute the code in the next code cell.
- Input the number of the image you want to process when prompted by the code during execution.
---

## Files and Descriptions

### main.cu

The `main.cu` file is the entry point for the project, orchestrating the sequence of operations across the CNN layers. It initializes layer structures, manages data flow, and invokes kernel functions for each layer. Key steps in the file include:

- **Image Indexing**: Loads the specified image for processing.
- **Layer Initialization**: Defines the dimensions and data for each CNN layer.
- **Layer-wise Operations**: 
  - Performs convolution using the `perform_convolution` kernel.
  - Executes max pooling and dense layer operations.
  
The program concludes by verifying the output against precomputed results.

---

### load_npy.cpp

The `load_npy.cpp` file contains functions for loading and managing `.npy` files, which store weights, biases, input data, and other layer-specific information. Key functions include:

- `Load_image`: Loads image data into the input layer.
- `populate_input_with_maps`: Copies data between layers for processing.
- `populate_verification`: Loads verification data for output validation.

These functions facilitate smooth data flow across layers by managing inputs and outputs efficiently.

---

### utils.cpp

This file contains utility functions for:

- **Verifying Outputs**: Compares kernel outputs against verification data.
- **Printing Data**: Prints layer data, including weights, biases, and outputs, for debugging purposes.
- **Testing**: Verifies the structure and dimensions of each layer before processing.

Example functions include `verify_kernel_output` for floating-point verification and `print_output` for formatted output display.

---

### Kernels.cu

This file contains the core CUDA kernels responsible for executing the CNN operations. These include:

- **Convolution_2D**: Performs 2D convolution on input data.
- **MaxPooling_2D**: Applies max pooling with defined window size and stride.
- **Dense**: Executes the fully connected layer's forward pass, including softmax activation.

Each kernel is optimized for GPU execution, and associated host functions (`perform_convolution`, `perform_max_pooling`, `perform_Dense`) handle memory allocation and data transfer between host and device.

---

### colab_notebook.ipynb

In this notebook, we perform the following steps:

### 1. Cloning Repositories
Clone the `Cuda_Convolution` and `cnpy` repositories, then build the `cnpy` library:
```bash
!git clone https://github.com/Shameerisb/Cuda_Convolution.git
!git clone https://github.com/rogersce/cnpy.git
!cd cnpy && mkdir build && cd build && cmake .. && make
```

### 2. Downloading and Preprocessing MNIST Data
Load and normalize the MNIST dataset, saving it as `.npy` files:
```python
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist

input_index = int(input("Please enter the input index: "))
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

np.save('/content/Cuda_Convolution/Data/Input_Data', X_train)
print("MNIST Data downloaded and saved")
```

### 3. Saving Model Weights and Biases
Load a pre-trained Keras model and save weights, biases, and shapes:
```python
from tensorflow.keras.models import load_model
import os

model = load_model('/content/Cuda_Convolution/Data/Model/mnist_model_Basic.keras')

def remove_none_shape(shape):
    return [dim for dim in shape if dim is not None]

save_dir = '/content/Cuda_Convolution/Data/Mnist_images/Weights_and_Biases'

for i, layer in enumerate(model.layers):
    i += 1
    layer_weights = layer.get_weights()
    input_shape = remove_none_shape(layer.input.shape) if layer.input is not None else []
    output_shape = remove_none_shape(layer.output.shape) if layer.output is not None else []

    if input_shape:
        np.save(os.path.join(save_dir, f'layer_{i}_input_shape.npy'), np.array(input_shape, dtype=np.int64))
    if output_shape:
        np.save(os.path.join(save_dir, f'layer_{i}_output_shape.npy'), np.array(output_shape, dtype=np.int64))

    if layer_weights:
        weights, biases = layer_weights
        if i < 4:
            weights = np.transpose(weights, (3, 0, 1, 2))

        np.save(os.path.join(save_dir, f'layer_{i}_weights.npy'), weights)
        np.save(os.path.join(save_dir, f'layer_{i}_biases.npy'), biases)

print("Weights, biases, and shapes saved.")
```

### 4. Saving Activation Maps
Create a model to output activations, run it, and save the results:
```python
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

input_layer = Input(shape=(28, 28, 1))
x = input_layer
layer_outputs = []

for layer in model.layers:
    x = layer(x)
    layer_outputs.append(x)

activation_model = Model(inputs=input_layer, outputs=layer_outputs)
sample_input = X_train[input_index].reshape(1, 28, 28, 1)
activations = activation_model.predict(sample_input)

base_path = '/content/Cuda_Convolution/Data/Mnist_images/Verification_data/'

for layer_index, activation in enumerate(activations):
    activation_maps = np.squeeze(activation)
    np.save(f"{base_path}Input_{input_index}_L_{layer_index + 1}.npy", activation_maps)

print("Activation maps saved successfully.")
```

### 5. Compiling the CUDA Files
Compile CUDA files while linking against the `cnpy` library:
```bash
!nvcc -o Mnist_1st_layer \
'/content/cnpy/cnpy.cpp' \
'/content/Cuda_Convolution/Source_Code/Mnist_main.cu' \
'/content/Cuda_Convolution/Source_Code/utils.cpp' \
'/content/Cuda_Convolution/Source_Code/load_npy.cpp' \
'/content/Cuda_Convolution/Source_Code/Kernels.cu' \
-I/usr/include \
-I/usr/local/cuda/include \
-L/usr/lib/x86_64-linux-gnu \
-lcudnn -lcuda -Icnpy -lz
```

### 6. Running the CUDA Executable
Run the compiled program, with optional profiling:
```bash
profiling = input("Do you want to perform profiling? (y/n): ")

if profiling == 'y':
    !nvprof --print-gpu-trace ./Mnist_1st_layer
else:
    !./Mnist_1st_layer
```


---

## Compilation Instructions

To compile the project in Colab, use the following command:

```bash
!nvcc -o Mnist_1st_layer  \
'/content/cnpy/cnpy.cpp' \
'/content/Cuda_Convolution/Source_Code/Mnist_main.cu' \
'/content/Cuda_Convolution/Source_Code/utils.cpp' \
'/content/Cuda_Convolution/Source_Code/load_npy.cpp' \
'/content/Cuda_Convolution/Source_Code/Kernels.cu' \
-I/usr/include \
-I/usr/local/cuda/include
```