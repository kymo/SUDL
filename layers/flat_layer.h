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

#ifndef FLATTERN_LAYER_H
#define FLATTERN_LAYER_H

#include "layer.h"
#include "full_conn_layer.h"

namespace sub_dl {

class FlatternLayer : public Layer {

public:
    FlatternLayer() {
        _type = FLAT;    
    }
    ~FlatternLayer() {}

    matrix_double _flatten(const std::vector<matrix_double>& data) {
        
        if (data.size() == 0) {
            std::cerr << "Flatten error!" << std::endl;
            exit(1);
        }
        int data_size = data.size();
        int data_x_dim = data[0]._x_dim;
        int data_y_dim = data[0]._y_dim;
        matrix_double t_matrix(1, data_size * data_x_dim * data_y_dim);
        for (int i = 0; i < t_matrix._y_dim; i++) {
            int n = i / (data_x_dim * data_y_dim);
            int inner_n = i - n * (data_x_dim * data_y_dim);
            t_matrix[0][i] = data[n][inner_n / data_y_dim][inner_n % data_y_dim]; 
        }
        return t_matrix;

    }

    void _forward(Layer* pre_layer) {
        std::vector<matrix_double>().swap(_data);
        _data.push_back(_flatten(pre_layer->_data));
        _pre_layer = pre_layer;
    }

    void _backward(Layer* nxt_layer) {
        std::vector<matrix_double>().swap(_errors);
        if (nxt_layer->_type != FULL_CONN) {
            exit(1);
        }
        const FullConnLayer* full_conn_layer = (FullConnLayer*)(nxt_layer);
        _errors.push_back(full_conn_layer->_errors[0] * full_conn_layer->_full_conn_weights._T());
    }
    
    void display() {

    }
    
    void _update_gradient(int opt_type, double learning_rate) {
    }


};
}

#endif
