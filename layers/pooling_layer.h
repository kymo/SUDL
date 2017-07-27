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

#ifndef POOLING_LAYER_H
#define POOLING_LAYER_H

#include "layer.h"

namespace sub_dl {

class PoolingLayer : public Layer {

public:
    int _pooling_x_dim;
    int _pooling_y_dim;
    int _pooling_type; // 0 max_pooling 1 avg_pooling
    matrix_double _pooling_weights;
    matrix_double _pooling_bias;
    matrix_double _delta_pooling_weights;
    matrix_double _delta_pooling_bias;
    virtual ~ PoolingLayer() {}
    
    PoolingLayer() {}

    PoolingLayer(int input_dim, int output_dim, 
        int pooling_x_dim, int pooling_y_dim, 
        int feature_x_dim, int feature_y_dim);

    void display();
    
    void _forward(Layer* pre_layer);
    
    void _backward(Layer* _nxt_layer);

	void _update_gradient(int opt_type, double learning_rate) {
	}
};

}
#endif
