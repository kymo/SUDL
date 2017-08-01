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

#ifndef RNN_CELL_H
#define RNN_CELL_H

#include "layer.h"

namespace sub_dl {

class RnnCell : public Layer {

public:
    // input to hidden node weights
    matrix_double _input_hidden_weights;
    // weights between hidden nodes of conjuctive time
    matrix_double _hidden_weights;
    // bias of hidden node
    matrix_double _hidden_bias;
    
    // gradient of _input_hidden_weights
    matrix_double _delta_input_hidden_weights;
    // gradient of _hidden_weights
    matrix_double _delta_hidden_weights;
    // gradient of _hidden_bias
    matrix_double _delta_hidden_bias;

    // pre layer data 
    std::vector<matrix_double> _pre_layer_data;
    double _eta;
    double _clip_gra;

    RnnCell(int input_dim, int output_dim);

    /* 
    * @brief forward function fo basic rnn cell with tanh
    *
    * @param
    *    pre_layer: layer before rnn_cell
    *        the type of legal layers are:
    *            RNN_CELL INPUT LSTM_CELL GRU_CELL
    *
    * @return
    *    void
    *
    */
    void _forward(Layer* pre_layer);

    void _backward(Layer* nxt_layer);

    void display() {}

    void _update_gradient(int opt_type, double learning_rate);

    void _clear_gradient() {
        _delta_input_hidden_weights.resize(0.0);
        _delta_hidden_weights.resize(0.0);
        _delta_hidden_bias.resize(0.0);
    }
};

}

#endif
