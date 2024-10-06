#include "load_npy.h"
#include <iostream>
#include <vector>
#include <cstring>
#include <filesystem>
#include <iomanip>  // For std::fixed and std::setprecision
#include "/content/cnpy/cnpy.h"  // Ensure you have cnpy.h in your include path

using namespace std;
namespace fs = std::filesystem;

void Load_image(layer_data &L_DATA, int image_index) {

  string base_path = "/content/Cuda_Convolution/Data/Mnist_images/Weights_and_Biases";
  string input_file = "/content/Cuda_Convolution/Data/Input_Data.npy";


  // Load input data from a .npy file
  cnpy::NpyArray input_npy = cnpy::npy_load(input_file);

  // Get the shape of the input data
  vector<size_t> input_shape = input_npy.shape;

  // Calculate the size of the specific image based on its shape
  size_t size = input_shape[1] * input_shape[2];

  // Allocate memory for the input data
  L_DATA.h_input = new float[size];

  // Get pointer to the input data
  float* input_data = input_npy.data<float>();

  // Copy the specific image data to h_input
  memcpy(L_DATA.h_input, &input_data[image_index * size], size * sizeof(float));

}

void populate_input_with_maps(layer_data &L_DATA, layer_DIM &L_DIM, layer_data &L_DATA_input) {
    int size = L_DIM.input.x * L_DIM.input.y * L_DIM.input.z + 1;
    int size_bytes = size * sizeof(float);
    
    // Copy data
    memcpy(L_DATA.h_input, L_DATA_input.h_output, size_bytes);

}


int populate_verification(layer_data &L_DATA, int L_index, int image_index) {
    
  string v_file = "/content/Cuda_Convolution/Data/Mnist_images/Verification_data/Input_" + to_string(image_index) + "_L_" + to_string(L_index) + ".npy";

  try {
  // Load the .npy file using cnpy
  cnpy::NpyArray v_npy = cnpy::npy_load(v_file);


  // Ensure the data type is float
  assert(v_npy.word_size == sizeof(float));

  // Get the shape of the data
  vector<size_t> v_shape = v_npy.shape;

  size_t size = 0;

  // Calculate the number of elements
  if (v_shape.size() == 3){
  size = v_shape[0] * v_shape[1] * v_shape[2];
  }
  else if (v_shape.size() == 2){
  size = v_shape[0] * v_shape[1];
  }
  else {
  size = v_shape[0];
  }


  // Allocate memory for v_output
  L_DATA.v_output = new float[size];

  float* v_data = v_npy.data<float>();

  // Copy the data to v_output
  memcpy(L_DATA.v_output, v_data, size * sizeof(float));

  return 0;
  } catch (...) {
    cout << "Save the respective verification files using \"Download Mnist and save verification files code\" cell for the image index " << image_index << endl;
    return 1;  // Exit if the file can't be loaded
  }
  
}



int populate_input_layer(layer_DIM &L_DIM, layer_data &L_DATA, int L_index, int image_index) {
    

    string base_path = "/content/Cuda_Convolution/Data/Mnist_images/Weights_and_Biases/";
    string weights_file = base_path + "layer_" + to_string(L_index) + "_weights.npy";
    string biases_file = base_path + "layer_" + to_string(L_index) + "_biases.npy";
    string input_shape_file = base_path + "layer_" + to_string(L_index) + "_input_shape.npy";
    string output_shape_file = base_path + "layer_" + to_string(L_index) + "_output_shape.npy";

    cnpy::NpyArray input_shape_npy = cnpy::npy_load(input_shape_file);
    cnpy::NpyArray output_shape_npy = cnpy::npy_load(output_shape_file);

    vector<size_t> input_shape = input_shape_npy.as_vec<size_t>();
    vector<size_t> output_shape = output_shape_npy.as_vec<size_t>();

    // Populate layer_DIM
    L_DIM.index = L_index;
    L_DATA.index = L_index;

    if (input_shape.size() == 3){

      L_DIM.input.x = input_shape[0];
      L_DIM.input.y = input_shape[1];
      L_DIM.input.z = input_shape[2];
    }
    else if (input_shape.size() == 2){
      L_DIM.input.x = input_shape[0];
      L_DIM.input.y = input_shape[1];
      L_DIM.input.z = 1;
    }
    else {
      L_DIM.input.x = input_shape[0];
      L_DIM.input.y = 1;
      L_DIM.input.z = 1;
    }

    if (output_shape.size() == 3){

      L_DIM.output.x = output_shape[0];
      L_DIM.output.y = output_shape[1];
      L_DIM.output.z = output_shape[2];
    }
    else if (output_shape.size() == 2){
      L_DIM.output.x = output_shape[0];
      L_DIM.output.y = output_shape[1];
      L_DIM.output.z = 1;
    }
    else {
      L_DIM.output.x = output_shape[0];
      L_DIM.output.y = 1;
      L_DIM.output.z = 1;
    }


    // Check if files exist
    bool weights_exist = fs::exists(weights_file);
    bool biases_exist = fs::exists(biases_file);

    // Declare variables for weights, biases, and their shapes outside the if block
    float* weights = nullptr;
    float* biases = nullptr;
    vector<size_t> weights_shape;
    vector<size_t> biases_shape;

    if (weights_exist && biases_exist) {
        // Load data
        cnpy::NpyArray weights_npy = cnpy::npy_load(weights_file);
        cnpy::NpyArray biases_npy = cnpy::npy_load(biases_file);

        weights = weights_npy.data<float>();
        biases = biases_npy.data<float>();

        weights_shape = weights_npy.shape;
        biases_shape = biases_npy.shape;

        L_DIM.bias = biases_shape[0];

        if(weights_shape.size() == 4){
          cout << "Loading weights for Convolution layer x, y, z, num_filters" << endl;
          L_DIM.mask.x = weights_shape[1];
          L_DIM.mask.y = weights_shape[2];
          L_DIM.mask.z = weights_shape[3];
          L_DIM.mask.num_filters = weights_shape[0];
        }
        else {
          cout << weights_shape[0] << endl;
          cout << "Loading weights for Dense layer x, y" << endl;
          L_DIM.mask.x = weights_shape[0];
          L_DIM.mask.y = weights_shape[1];
          L_DIM.mask.z = 1;
          L_DIM.mask.num_filters = 1;
        }

        // Initialize layer data
        L_DATA.initialize(L_DIM);

        // Copy data to layer_data
        memcpy(L_DATA.h_mask, weights, L_DIM.mask.x * L_DIM.mask.y * L_DIM.mask.z * L_DIM.mask.num_filters * sizeof(float));
        memcpy(L_DATA.h_bias, biases, biases_shape[0] * sizeof(float));
    } else {
        // If weights or biases do not exist, initialize them with zeros
        L_DIM.mask.x = 1;
        L_DIM.mask.y = 1;
        L_DIM.mask.z = 1;
        L_DIM.mask.num_filters = 1;
        L_DATA.initialize(L_DIM);

    }

    if (populate_verification(L_DATA, L_index, image_index)){
      return 1;
    } else {
      return 0;
    }
}


void populate_layer(layer_DIM &L_DIM, layer_data &L_DATA, layer_data &L_DATA_input, int L_index ,int image_index){

  populate_input_layer(L_DIM, L_DATA, L_index, image_index);
  populate_input_with_maps(L_DATA, L_DIM, L_DATA_input);

}