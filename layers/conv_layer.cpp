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

#include "conv_layer.h"
#include "pooling_layer.h"
#include "full_conn_layer.h"

namespace sub_dl {

ConvLayer::ConvLayer(int input_dim, int output_dim, 
    int kernel_x_dim, int kernel_y_dim, 
    int feature_x_dim, int feature_y_dim) {

    _input_dim = input_dim;
    _output_dim = output_dim;
    _kernel_x_dim = kernel_x_dim;
    _kernel_y_dim = kernel_y_dim;
    _feature_x_dim = feature_x_dim;
    _feature_y_dim = feature_y_dim;

    _feature_x_dim = feature_x_dim;
    _feature_y_dim = feature_y_dim;

    _conv_kernels.resize(_input_dim, _output_dim);
    _delta_conv_kernels.resize(_input_dim, _output_dim);
    for (int i = 0; i < _input_dim;i ++) {
        for (int j = 0; j < _output_dim; j++) {    
            matrix_double kernel(_kernel_x_dim, _kernel_y_dim);
            kernel.assign_val();
            _conv_kernels[i][j] = kernel;
            _delta_conv_kernels[i][j].resize(_kernel_x_dim, _kernel_y_dim);
        }
    }
    _conv_bias.resize(1, _output_dim);
    _conv_bias.assign_val();
    _delta_conv_bias.resize(1, _output_dim);
    _conn_map.resize(_input_dim, _output_dim);
    _conn_map.resize(1);
    _type = CONV;

}

void ConvLayer::display() {}

void ConvLayer::_forward(Layer* pre_layer) {
    // save the pre layer pointer for backward weight update
    _pre_layer = pre_layer;
    std::vector<matrix_double>().swap(_data);
    for (int i = 0; i < _output_dim; i++) {
        matrix_double feature_map(_feature_x_dim, _feature_y_dim);
        for (int j = 0; j < _input_dim; j++) {
            if (_conn_map[j][i]) {
                feature_map = feature_map + pre_layer->_data[j].conv(_conv_kernels[j][i]);
            }
        }
        _data.push_back(feature_map + _conv_bias[0][i]);
    }
}

void ConvLayer::_backward(Layer* nxt_layer) {
    if (nxt_layer->_type != ACT) {
        FATAL_LOG("Wrong layer type after convonv layer! func[%s] line[%d]", __func__, __LINE__);
        exit(1);
    }
    std::vector<matrix_double>().swap(_errors);
    if (nxt_layer->_type == POOL) {
        const PoolingLayer* pooling_layer = (PoolingLayer*)(nxt_layer);
        int delta_size = (pooling_layer->_pooling_x_dim) * pooling_layer->_pooling_y_dim;
        for (int i = 0; i < _output_dim; i++) {
            matrix_double up_delta = pooling_layer->_errors[i]
                .up_sample(pooling_layer->_pooling_x_dim, 
                pooling_layer->_pooling_y_dim);
            matrix_double error = up_delta.dot_mul(sigmoid_m_diff(_data[i]))
                * (pooling_layer->_pooling_weights[0][i] / delta_size);
            _errors.push_back(error);
        }
    } else if (nxt_layer->_type == ACT) {
        _errors = nxt_layer->_errors;    
    }

    for (int i = 0; i < _output_dim; i++) {
        // update kernel weights
        for (int j = 0; j < _input_dim; j++) {
            if (_conn_map[j][i]) {
                _delta_conv_kernels[j][i] = (_pre_layer->_data[j]
                    .conv(_errors[i].rotate_180()))
                    .rotate_180();
            }
        }
        // update kernel bias
        _delta_conv_bias[0][i] = _errors[i].sum();
    }
}

void ConvLayer::_update_gradient(int opt_type, double learning_rate) {
    if (opt_type == SGD) {
        _conv_kernels.add(_delta_conv_kernels * learning_rate);
        _conv_bias.add(_delta_conv_bias * learning_rate);
    }
}

}
