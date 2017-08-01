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

#include "layer.h"
#include "lstm_cell.h"
#include "rnn_cell.h"
#include "gru_cell.h"
#include "seq_full_conn_layer.h"

namespace sub_dl {

GruCell::GruCell(int input_dim, int output_dim) {
    _input_dim = input_dim;
    _output_dim = output_dim;

    _type = GRU_CELL;
    
    _ug_input_weights.resize(_input_dim, _output_dim);
    _ug_hidden_weights.resize(_output_dim, _output_dim);
    _ug_bias.resize(1, _output_dim);
    
    _rg_input_weights.resize(_input_dim, _output_dim);
    _rg_hidden_weights.resize(_output_dim, _output_dim);
    _rg_bias.resize(1, _output_dim);

    _newh_input_weights.resize(_input_dim, _output_dim);
    _newh_hidden_weights.resize(_output_dim, _output_dim);
    _newh_bias.resize(1, _output_dim);
    
    _ug_input_weights.assign_val();
    _ug_hidden_weights.assign_val();
    _ug_bias.assign_val();
    
    _rg_input_weights.assign_val();
    _rg_hidden_weights.assign_val();
    _rg_bias.assign_val();

    _newh_input_weights.assign_val();
    _newh_hidden_weights.assign_val();
    _newh_bias.assign_val();
    
    _delta_ug_input_weights.resize(_input_dim, _output_dim);
    _delta_ug_hidden_weights.resize(_output_dim, _output_dim);
    _delta_ug_bias.resize(1, _output_dim);
    _delta_rg_input_weights.resize(_input_dim, _output_dim);
    _delta_rg_hidden_weights.resize(_output_dim, _output_dim);
    _delta_rg_bias.resize(1, _output_dim);
    _delta_newh_input_weights.resize(_input_dim, _output_dim);
    _delta_newh_hidden_weights.resize(_output_dim, _output_dim);
    _delta_newh_bias.resize(1, _output_dim);
    
}

void GruCell::_forward(Layer* pre_layer) {
    std::vector<matrix_double>().swap(_data);
    _seq_len = pre_layer->_seq_len;
    matrix_double pre_hidden_vals(1, _output_dim);

    _ug_values.resize(_seq_len, _output_dim);
    _rg_values.resize(_seq_len, _output_dim);
    _newh_values.resize(_seq_len, _output_dim);
    _hidden_values.resize(_seq_len, _output_dim);

    for (int t = 0; t < _seq_len; t++) { 
        const matrix_double& xt = pre_layer->_data[t];
        matrix_double ug_value = sigmoid_m(xt * _ug_input_weights
            + pre_hidden_vals * _ug_hidden_weights 
            + _ug_bias);
        matrix_double rg_value = sigmoid_m(xt * _rg_input_weights
            + pre_hidden_vals * _rg_hidden_weights
            + _rg_bias);
        matrix_double newh_value = tanh_m(xt * _newh_input_weights
            + (rg_value.dot_mul(pre_hidden_vals)) * _newh_hidden_weights
            + _newh_bias);
        matrix_double hidden_value = ug_value.dot_mul(newh_value) - (ug_value - 1).dot_mul(pre_hidden_vals) ;

        pre_hidden_vals = hidden_value;
        _hidden_values.set_row(t, hidden_value);
        _ug_values.set_row(t, ug_value);
        _rg_values.set_row(t, rg_value);
        _newh_values.set_row(t, newh_value);
        _data.push_back(hidden_value);
    }
    _pre_layer = pre_layer;
    _pre_layer_data = pre_layer->_data;
}

void GruCell::_backward(Layer* nxt_layer) {
    if (nxt_layer->_type != SEQ_FULL 
        && nxt_layer->_type != RNN_CELL
        && nxt_layer->_type != LSTM_CELL
        && nxt_layer->_type != GRU_CELL) {
        FATAL_LOG("Layer before gru cell is not legal func[%s] line[%d]", __func__, __LINE__);
        exit(1);
    }

    matrix_double nxt_hidden_error(1, _output_dim);
    matrix_double nxt_newh_error(1, _output_dim);
    matrix_double nxt_ug_error(1, _output_dim);
    matrix_double nxt_rg_error(1, _output_dim);
    
    std::vector<matrix_double> nxt_layer_error_weights;

    std::vector<matrix_double>().swap(_rg_errors);
    std::vector<matrix_double>().swap(_ug_errors);
    std::vector<matrix_double>().swap(_newh_errors);

    if (nxt_layer->_type == SEQ_FULL) {
        SeqFullConnLayer* seq_full_layer = (SeqFullConnLayer*) nxt_layer;
        for (int i = 0; i < _seq_len; i++) {
            nxt_layer_error_weights.push_back(nxt_layer->_errors[i]
                * seq_full_layer->_seq_full_weights._T());
        }
    } else if (nxt_layer->_type == RNN_CELL) {
        RnnCell* rnn_cell = (RnnCell*) nxt_layer;
        for (int i = 0; i < _seq_len; i++) {
            nxt_layer_error_weights.push_back(rnn_cell->_errors[i] 
                * rnn_cell->_input_hidden_weights._T());
        }
    } else if (nxt_layer->_type == LSTM_CELL) {
        LstmCell* lstm_cell = (LstmCell*) nxt_layer;
        for (int i = 0; i < _seq_len; i++) {
            nxt_layer_error_weights.push_back(lstm_cell->_fg_errors[i] * lstm_cell->_fg_input_weights._T()
                + lstm_cell->_ig_errors[i] * lstm_cell->_ig_input_weights._T()
                + lstm_cell->_og_errors[i] * lstm_cell->_og_input_weights._T()
                + lstm_cell->_new_cell_errors[i] * lstm_cell->_cell_input_weights._T());
        }
    } else if (nxt_layer->_type == GRU_CELL) {
        GruCell* gru_cell = (GruCell*) nxt_layer;
        for (int i = 0; i < _seq_len; i++) {
            nxt_layer_error_weights.push_back(gru_cell->_ug_errors[i] * gru_cell->_ug_input_weights._T()
                + gru_cell->_rg_errors[i] * gru_cell->_rg_input_weights._T()
                + gru_cell->_newh_errors[i]  * gru_cell->_newh_input_weights._T());
        }
    }

    for (int t = _seq_len - 1; t >= 0; t--) {
        
        // before get the output gate/input gate/forget gate error
        // the mid error value should be calculated first
        // cell_mid_error and hidden_mid_error
        matrix_double mid_hidden_error = nxt_layer_error_weights[t] 
            + nxt_ug_error * _ug_hidden_weights._T() 
            + nxt_rg_error * _rg_hidden_weights._T();
    
        if (t + 1 < _seq_len) {
            mid_hidden_error = mid_hidden_error 
                + (nxt_newh_error * _newh_hidden_weights._T()).dot_mul(_rg_values._R(t + 1))
                - nxt_hidden_error.dot_mul(_ug_values._R(t + 1) - 1);
        }
        
        matrix_double newh_error = mid_hidden_error
            .dot_mul(_ug_values._R(t))
            .dot_mul(tanh_m_diff(_newh_values._R(t)));

        matrix_double ug_error = mid_hidden_error.dot_mul(sigmoid_m_diff(_ug_values._R(t)));
        if (t > 0) {
            ug_error = ug_error.dot_mul(_newh_values._R(t) - _hidden_values._R(t - 1));
        } else {
            ug_error = ug_error.dot_mul(_newh_values._R(t));
        }

        matrix_double rg_error(1, _output_dim);
        if (t > 0) {
            rg_error = _hidden_values._R(t - 1)
                .dot_mul(newh_error * _newh_hidden_weights._T())
                .dot_mul(sigmoid_m_diff(_rg_values._R(t)));
        }
        
        _delta_ug_input_weights.add(_pre_layer_data[t]._T() * ug_error);
        _delta_ug_bias.add(ug_error);
        _delta_rg_input_weights.add(_pre_layer_data[t]._T() * rg_error);
        _delta_rg_bias.add(rg_error);
        _delta_newh_input_weights.add(_pre_layer_data[t]._T() * newh_error);
        _delta_newh_bias.add(newh_error);
        
        if (t > 0) {
            _delta_ug_hidden_weights.add(_hidden_values._R(t - 1)._T() * ug_error);
            _delta_rg_hidden_weights.add(_hidden_values._R(t - 1)._T() * rg_error);
            _delta_newh_hidden_weights.add((_hidden_values._R(t - 1).dot_mul(_rg_values._R(t)))._T() * newh_error);
        }
        
        nxt_hidden_error = mid_hidden_error;
        nxt_ug_error = ug_error;
        nxt_rg_error = rg_error;
        nxt_newh_error = newh_error;
        _ug_errors.push_back(ug_error);
        _rg_errors.push_back(rg_error);
        _newh_errors.push_back(newh_error);
    }
    std::reverse(_ug_errors.begin(), _ug_errors.end());
    std::reverse(_rg_errors.begin(), _rg_errors.end());
    std::reverse(_newh_errors.begin(), _newh_errors.end());

}

void GruCell::_update_gradient(int opt_type, double learning_rate) {
    if (opt_type == SGD) {
        _ug_input_weights.add(_delta_ug_input_weights * learning_rate);
        _rg_input_weights.add(_delta_rg_input_weights * learning_rate);
        _newh_input_weights.add(_delta_newh_input_weights * learning_rate);
        _ug_hidden_weights.add(_delta_ug_hidden_weights * learning_rate);
        _rg_hidden_weights.add(_delta_rg_hidden_weights * learning_rate);
        _newh_hidden_weights.add(_delta_newh_hidden_weights * learning_rate);
        _ug_bias.add(_delta_ug_bias * learning_rate);
        _rg_bias.add(_delta_rg_bias * learning_rate);
        _newh_bias.add(_delta_newh_bias * learning_rate);
    }
}

}
