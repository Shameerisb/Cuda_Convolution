

#ifndef LOAD_NPY_H
#define LOAD_NPY_H

#include <iostream>
#include <vector>
#include <string>
#include "/content/cnpy/cnpy.h"

using namespace std;

struct size_input {
    int x, y, z;

    // Default constructor
    size_input() : x(0), y(0), z(0) {}

    // Parameterized constructor
    size_input(int x, int y, int z) : x(x), y(y), z(z) {}
};

struct size_mask {
    int x, y, z, num_filters;

    // Default constructor
    size_mask() : x(0), y(0), z(0), num_filters(0) {}

    // Parameterized constructor
    size_mask(int x, int y, int z, int num_filters) : x(x), y(y), z(z), num_filters(num_filters) {}
};

struct size_output {
    int x, y, z;

    // Default constructor
    size_output() : x(0), y(0), z(0) {}

    // Parameterized constructor
    size_output(int x, int y, int z) : x(x), y(y), z(z) {}
};

struct layer_DIM {

  int index;
  size_input input;
  size_mask mask;
  size_output output;
  int bias;


  // Default constructor
  layer_DIM() : input(), mask(), output(), index(0), bias(0) {}

  // Parameterized constructor
  layer_DIM(int i_x, int i_y, int i_z, int m_x, int m_y, int m_z, int o_x, int o_y, int o_z, int filters, int bias, int index)
      : input(i_x, i_y, i_z), mask(m_x, m_y, m_z, filters), output(o_x, o_y, o_z), bias(bias), index(index) {}
};

// Structure for layer data
struct layer_data {
  
  int index;
  float *h_input;
  float *h_mask;
  float *h_bias;
  float *h_output;
  float *v_output;

  // Default constructor
  layer_data() : h_input(nullptr), h_mask(nullptr), h_bias(nullptr), h_output(nullptr), v_output(nullptr), index(0) {}

  // Method to initialize memory for layer data
  void initialize(const layer_DIM& L_DIM) {
      // Calculate total size needed for each array
      int input_total_size = L_DIM.input.x * L_DIM.input.y * L_DIM.input.z;
      int output_total_size = L_DIM.output.x * L_DIM.output.y * L_DIM.output.z;
      int mask_total_size = L_DIM.mask.x * L_DIM.mask.y * L_DIM.mask.z * L_DIM.mask.num_filters;
      int bias_total_size = L_DIM.bias;

      // Allocate memory for 3D arrays as contiguous blocks
      h_input = new float[input_total_size];
      h_output = new float[output_total_size];
      v_output = new float[output_total_size];
      h_mask = new float[mask_total_size];
      h_bias = new float[bias_total_size];

      // Initialize memory to zero
      // std::fill(h_input, h_input + input_total_size, 0.0);
      std::fill(h_output, h_output + output_total_size, 0.0);
      std::fill(v_output, v_output + output_total_size, 0.0);
      std::fill(h_mask, h_mask + mask_total_size, 0.0);
      std::fill(h_bias, h_bias + bias_total_size, 0.0);
  }
};


// Function to populate layer information

void Load_image(layer_data &L_DATA, int image_index);
void populate_input_with_maps(layer_data &L_DATA, layer_DIM &L_DIM, layer_data &L_DATA_input);

int populate_verification (layer_data &L_DATA, int L_index, int image_index);

int populate_input_layer(layer_DIM &L_DIM, layer_data &L_DATA, int L_index, int image_index);
void populate_layer(layer_DIM &L_DIM, layer_data &L_DATA, layer_data &L_DATA_input, int L_index ,int image_index);

#endif // LOAD_NPY_H
