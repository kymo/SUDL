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

    RnnCell(int input_dim, int output_dim) {
        
        _input_dim = input_dim;
        _output_dim = output_dim;    
        _type = RNN_CELL;
        _input_hidden_weights.resize(_input_dim, _output_dim);
        _hidden_weights.resize(_output_dim, _output_dim);
        _hidden_bias.resize(1, _output_dim);

        _input_hidden_weights.assign_val();
        _hidden_weights.assign_val();
        _hidden_bias.assign_val();

    }

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
    void _forward(Layer* pre_layer) {
        std::vector<matrix_double>().swap(_data);
        _seq_len = pre_layer->_data.size();
        matrix_double pre_hidden_vals(1, _output_dim);
        for (int t = 0; t < _seq_len; t++) { 
            const matrix_double& xt = pre_layer->_data[t];
            matrix_double net_h_vals = xt * _input_hidden_weights + 
                pre_hidden_vals * _hidden_weights + _hidden_bias;
            pre_hidden_vals = tanh_m(net_h_vals);
            _data.push_back(pre_hidden_vals);
        }
        _pre_layer = pre_layer;
        _pre_layer_data = pre_layer->_data;
    }

    void _backward(Layer* nxt_layer) {
        if (nxt_layer->_type != SEQ_FULL && nxt_layer->_type != RNN_CELL) {
            exit(1);
        }
        std::vector<matrix_double>().swap(_errors);
        matrix_double nxt_hidden_error(1, _output_dim);
        
        _delta_input_hidden_weights.resize(_input_dim, _output_dim);
        _delta_hidden_weights.resize(_output_dim, _output_dim);
        _delta_hidden_bias.resize(1, _output_dim);
        matrix_double pre_layer_weights;
        if (nxt_layer->_type == SEQ_FULL) {
            SeqFullConnLayer* seq_full_layer = (SeqFullConnLayer*) nxt_layer;
            pre_layer_weights =  seq_full_layer->_seq_full_weights;
        } else if (nxt_layer->_type == RNN_CELL) {
            RnnCell* rnn_cell = (RnnCell*) nxt_layer;
            pre_layer_weights = rnn_cell->_input_hidden_weights;
        }

        for (int t = _seq_len - 1; t >= 0; t--) {
            matrix_double hidden_error = (nxt_layer->_errors[t] * pre_layer_weights._T()
                + nxt_hidden_error * _hidden_weights._T())
                .dot_mul(tanh_m_diff(_data[t]));
            _delta_input_hidden_weights.add(_pre_layer_data[t]._T() * hidden_error);
            if (t > 0) {
                _delta_hidden_weights.add(_data[t - 1]._T() * hidden_error);
            }
            _delta_hidden_bias.add(hidden_error);
            nxt_hidden_error = hidden_error;
            _errors.push_back(hidden_error);            
        }
        std::reverse(_errors.begin(), _errors.end());
    }

    void display() {}

    void _update_gradient(int opt_type, double learning_rate) {
        if (opt_type == SGD) {
            _input_hidden_weights.add(_delta_input_hidden_weights * learning_rate);
            _hidden_bias.add(_delta_hidden_bias * learning_rate);
            _hidden_weights.add(_delta_hidden_weights * learning_rate);
        }
    }

};

}

#endif
