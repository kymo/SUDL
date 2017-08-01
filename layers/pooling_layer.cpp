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

#include "pooling_layer.h"
#include "conv_layer.h"

namespace sub_dl {

PoolingLayer::PoolingLayer(int input_dim, int output_dim, 
    int pooling_x_dim, int pooling_y_dim,
    int feature_x_dim, int feature_y_dim) {

    _input_dim = input_dim;
    _output_dim = output_dim;
    _pooling_x_dim = pooling_x_dim;
    _pooling_y_dim = pooling_y_dim;
    _feature_x_dim = feature_x_dim;
    _feature_y_dim = feature_y_dim;
    _type = POOL;

}

void PoolingLayer::display() {}

void PoolingLayer::_forward(Layer* pre_layer) {
    std::vector<matrix_double>().swap(_data);
    _pre_layer = pre_layer;
    for (int i = 0; i < _output_dim; i++) {
        matrix_double up_feature = (pre_layer->_data[i]
            .down_sample(_pooling_x_dim, _pooling_y_dim, AVG_POOLING)); 
        _data.push_back(up_feature);
    }
}

void PoolingLayer::_backward(Layer* nxt_layer) {
    if (nxt_layer->_type != ACT && nxt_layer->_type != CONV) {
        FATAL_LOG("Error layer type error before pooling! func[%s] line[%d]", __func__, __LINE__);
        exit(1);
    }

    std::vector<matrix_double>().swap(_errors);
    if (nxt_layer->_type == CONV) { 
        const ConvLayer* conv_layer = (ConvLayer*)(nxt_layer);
        for (int i = 0; i < _output_dim; i++) {
            matrix_double error(_feature_x_dim, _feature_y_dim);
            for (int j = 0; j < conv_layer->_output_dim; j++) {
                if (conv_layer->_conn_map[i][j]) {
                    matrix_double conv2d_vec = conv_layer->_errors[j]
                        .conv2d(conv_layer->_conv_kernels[i][j].rotate_180(), FULL);
                    error = error + conv2d_vec;
                }
            }
            _errors.push_back(error);
        }
    } else {
        _errors = nxt_layer->_errors;
    }
}

}
