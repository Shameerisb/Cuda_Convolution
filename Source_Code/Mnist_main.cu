#include <cassert>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>
#include <cuda.h>

#include "Kernels.h" // Include the new header file
#include "utils.h"
#include "load_npy.h"

using namespace std;

int main() {

  int image_indextoload = 2;

  // Define dimensions for layers
  layer_DIM First_layer_DIM;
  layer_data First_layer_data;

  layer_DIM Second_layer_DIM;
  layer_data Second_layer_data;

  layer_DIM Third_layer_DIM;
  layer_data Third_layer_data;
      
  layer_DIM Fourth_layer_DIM;
  layer_data Fourth_layer_data;

  layer_DIM Fifth_layer_DIM;
  layer_data Fifth_layer_data;

  layer_DIM Sixth_layer_DIM;
  layer_data Sixth_layer_data;



  //          --------------First Conv2D layer-------------- 
  populate_input_layer(First_layer_DIM, First_layer_data, 1, image_indextoload);
  Load_image(First_layer_data, image_indextoload); // Load after populating else it will be data will be erased to zeros.
 
  // print_test(First_layer_DIM, First_layer_data);
  perform_convolution(First_layer_DIM, First_layer_data);

  // // Perform CuDNN convolution
  // perform_cudnn_convolution(First_layer_DIM, First_layer_data);

  // verify_kernel_output(First_layer_data, First_layer_DIM);

  // cout << "\n======================GPU_Result============================" << endl ;
  // print_output(First_layer_data.h_output, First_layer_DIM, 1);

  // cout << "\n===================Tensor_Flow_Result=========================" << endl ;
  // print_output(First_layer_data.v_output, First_layer_DIM, 1);








  //          --------------Second Conv2D layer--------------
  populate_layer(Second_layer_DIM, Second_layer_data, First_layer_data, 2, image_indextoload);
  
  // print_test(Second_layer_DIM, Second_layer_data);
  perform_convolution(Second_layer_DIM, Second_layer_data);

  //          --------------Third Conv2D layer-------------- 
  populate_layer(Third_layer_DIM, Third_layer_data, Second_layer_data, 3, image_indextoload);
 
  // print_test(Third_layer_DIM, Third_layer_data);
  perform_convolution(Third_layer_DIM, Third_layer_data);


  //          --------------Fourth Max_pooling Layer--------------
  populate_layer(Fourth_layer_DIM, Fourth_layer_data, Third_layer_data, 4, image_indextoload);
  // print_test(Fourth_layer_DIM, Fourth_layer_data);
  dim3 poolSize(2, 2, 1);
  dim3 stride(2, 2, 1);
  perform_max_pooling(Fourth_layer_DIM, Fourth_layer_data, poolSize, stride);


  //          --------------Fifth Flatten layer-------------- (Array Already Flattened)
  // /*
  populate_layer(Fifth_layer_DIM, Fifth_layer_data, Fourth_layer_data, 5, image_indextoload);
  // print_test(Fifth_layer_DIM, Fifth_layer_data);
  perform_flatting(Fifth_layer_DIM, Fifth_layer_data, Fourth_layer_DIM);
  // */

  //          --------------Sixth Dense layer-------------- 
  populate_layer(Sixth_layer_DIM, Sixth_layer_data, Fourth_layer_data, 6, image_indextoload);
 
  print_test(Sixth_layer_DIM, Sixth_layer_data);

  perform_Dense(Sixth_layer_DIM, Sixth_layer_data);




  cout << "\n\n\nCompleted" << endl;
  return 0;
}
