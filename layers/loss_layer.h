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

#ifndef LOSS_LAYER_H
#define LOSS_LAYER_H

#include "layer.h"

namespace sub_dl {

class LossLayer : public Layer {

public:
    matrix_double _label;
    LossLayer() {
        _type = LOSS;
    }

    ~LossLayer() {
    }

	void _set_label(const matrix_double& label) {
		_label = label;
	}

    void _update_gradient(int opt_type, double learning_rate) {}

    void display() {}
};

// mean square loss layer
class MeanSquareLossLayer : public LossLayer {

public:

    MeanSquareLossLayer() : LossLayer() {

	}

    void _forward(Layer* pre_layer) {
        std::vector<matrix_double>().swap(_data);
        _pre_layer = pre_layer;
        if (pre_layer->_type != ACT) {
            FATAL_LOG("Error pre layer for mean square loss layer, func[%s] line[%d]", __func__, __LINE__);
            exit(1);
        }
        matrix_double minus_result = pre_layer->_data[0] - _label; 
        _data.push_back(minus_result.dot_mul(minus_result) * 0.5);
    }
    
    void _backward(Layer* nxt_layer) {
        std::vector<matrix_double>().swap(_errors);
        _errors.push_back(_pre_layer->_data[0] - _label);
    }

};

class CrossEntropyLossLayer : public LossLayer {

public:

    void _forward(Layer* pre_layer) {
        std::vector<matrix_double>().swap(_data);
        _pre_layer = pre_layer;
        if (pre_layer->_type != FULL_CONN) {
            FATAL_LOG("Error pre layer for mean square loss layer, func[%s] line[%d]", __func__, __LINE__);
            exit(1);
        }

        _data.push_back(_label.dot_mul(log_m(pre_layer->_data[0])) 
            + _label.minus_by(1).dot_mul(log_m(pre_layer->_data[0].minus_by(1))));

    }

    void _backward(Layer* nxt_layer) {

    }
};

}

#endif
