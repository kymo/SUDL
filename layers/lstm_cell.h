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

#ifndef LSTM_CELL_H
#define LSTM_CELL_H

#include "layer.h"

namespace sub_dl {

class LSTM_OUT {

public:
    matrix_double _hidden_values;
    matrix_double _cell_values;
    matrix_double _og_values;
    matrix_double _ig_values;
    matrix_double _fg_values;
    matrix_double _cell_new_values;
    
    void _resize(int time_step_cnt,
        int output_dim) {
        _hidden_values.resize(time_step_cnt, output_dim);
        _cell_values.resize(time_step_cnt, output_dim);
        _ig_values.resize(time_step_cnt, output_dim);
        _og_values.resize(time_step_cnt, output_dim);
        _fg_values.resize(time_step_cnt, output_dim);
        _cell_new_values.resize(time_step_cnt, output_dim);
    }
};

class LstmCell : public Layer {

public:
    
    // input gate
    matrix_double _ig_input_weights;
    matrix_double _ig_hidden_weights;
    matrix_double _ig_cell_weights;
    matrix_double _ig_bias;
    
    matrix_double _ig_delta_input_weights;
    matrix_double _ig_delta_hidden_weights;
    matrix_double _ig_delta_cell_weights;
    matrix_double _ig_delta_bias;

    // forget gate
    matrix_double _fg_input_weights;
    matrix_double _fg_hidden_weights;
    matrix_double _fg_cell_weights;
    matrix_double _fg_bias;
    matrix_double _fg_delta_input_weights;
    matrix_double _fg_delta_hidden_weights;
    matrix_double _fg_delta_cell_weights;
    matrix_double _fg_delta_bias;

    // output gate
    matrix_double _og_input_weights;
    matrix_double _og_hidden_weights;
    matrix_double _og_cell_weights;
    matrix_double _og_bias;
    matrix_double _og_delta_input_weights;
    matrix_double _og_delta_hidden_weights;
    matrix_double _og_delta_cell_weights;
    matrix_double _og_delta_bias;

    // new cell state
    matrix_double _cell_input_weights;
    matrix_double _cell_hidden_weights;
    matrix_double _cell_bias;
    matrix_double _cell_delta_input_weights;
    matrix_double _cell_delta_hidden_weights;
    matrix_double _cell_delta_bias;

    // errors
    std::vector<matrix_double> _fg_errors;
    std::vector<matrix_double> _ig_errors;
    std::vector<matrix_double> _og_errors;
    std::vector<matrix_double> _new_cell_errors;

    // lstm cell mid values
    LSTM_OUT _lstm_layer_values;
    
    // pre layer data to store the data of pre layer, not using 
    // pre_layer pointer to get the data because in bi-directional
    // cell, the input data (data of pre layer) needed to be reversed
    std::vector<matrix_double> _pre_layer_data;

    bool _use_peephole;
    double _eta;
    double _clip_gra;

    LstmCell(const lm::RnnCellParam& lstm_param);
	// int input_dim, int output_dim, bool use_peephole);

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
        _ig_delta_input_weights.resize(0.0);
        _ig_delta_hidden_weights.resize(0.0);
        _ig_delta_cell_weights.resize(0.0);
        _ig_delta_bias.resize(0.0);
        
        _fg_delta_input_weights.resize(0.0);
        _fg_delta_hidden_weights.resize(0.0);
        _fg_delta_cell_weights.resize(0.0);
        _fg_delta_bias.resize(0.0);
        
        _og_delta_input_weights.resize(0.0);
        _og_delta_hidden_weights.resize(0.0);
        _og_delta_cell_weights.resize(0.0);
        _og_delta_bias.resize(0.0);
        
        _cell_delta_input_weights.resize(0.0);
        _cell_delta_hidden_weights.resize(0.0);
        _cell_delta_bias.resize(0.0);
    }
};

}

#endif
