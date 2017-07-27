// Copyright (c) 2017 kymowind@gmail.com. All Rights Reserve.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//    http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License. 

#ifndef CONV_LAYER_H
#define CONV_LAYER_H

#include "layer.h"

namespace sub_dl {

class ConvLayer : public Layer {

public:
    int _kernel_x_dim;
    int _kernel_y_dim;
    
    matrix_int _conn_map;
    
    Matrix<matrix_double> _conv_kernels;
    matrix_double _conv_bias;
    
    Matrix<matrix_double> _delta_conv_kernels;
    matrix_double _delta_conv_bias;

    virtual ~ConvLayer() {}
    ConvLayer() {}

    ConvLayer(int input_dim, int output_dim, 
        int kernel_x_dim, int kernel_y_dim, 
        int feature_x_dim, int feature_y_dim); 
    
    void display();

    /*
    * @brief set the connection map between the input feature map 
    *        and the output feature of the convolutional layer
    *
    * @param conn_map
    *
    * @ret None
    */
    void _set_conn_map(const matrix_int& conn_map) {
        _conn_map = conn_map;
    }

    void _forward(Layer* pre_layer);

    void _backward(Layer* nxt_layer);
    
    void _update_gradient(int opt_type, double learning_rate);

};

}
#endif
