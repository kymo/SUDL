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

LstmCell::LstmCell(int input_dim, int output_dim, bool use_peephole) {
    _input_dim = input_dim;
    _output_dim = output_dim;

    _type = LSTM_CELL;

    _ig_input_weights.resize(_input_dim, _output_dim);
    _ig_hidden_weights.resize(_output_dim, _output_dim);
    _ig_bias.resize(1, _output_dim);

    _ig_input_weights.assign_val();
    _ig_hidden_weights.assign_val();
    _ig_bias.assign_val();

    _fg_input_weights.resize(_input_dim, _output_dim);
    _fg_hidden_weights.resize(_output_dim, _output_dim);
    _fg_bias.resize(1, _output_dim);

    _fg_input_weights.assign_val();
    _fg_hidden_weights.assign_val();
    _fg_bias.assign_val();
    
    _og_input_weights.resize(_input_dim, _output_dim);
    _og_hidden_weights.resize(_output_dim, _output_dim);
    _og_bias.resize(1, _output_dim);

    _og_input_weights.assign_val();
    _og_hidden_weights.assign_val();
    _og_bias.assign_val();

    _cell_input_weights.resize(_input_dim, _output_dim);
    _cell_hidden_weights.resize(_output_dim, _output_dim);
    _cell_bias.resize(1, _output_dim);

    _cell_input_weights.assign_val();
    _cell_hidden_weights.assign_val();
    _cell_bias.assign_val();

    _use_peephole = use_peephole;
    
    if (use_peephole) {
        _fg_cell_weights.resize(_output_dim, _output_dim);
        _og_cell_weights.resize(_output_dim, _output_dim);
        _ig_cell_weights.resize(_output_dim, _output_dim);
    }

}

void LstmCell::_forward(Layer* pre_layer) {
    std::vector<matrix_double>().swap(_data);
    _seq_len = pre_layer->_data.size();
    matrix_double pre_hidden_vals(1, _output_dim);
    matrix_double pre_cell_vals(1, _output_dim);
    _lstm_layer_values._resize(_seq_len, _output_dim);
    for (int t = 0; t < _seq_len; t++) { 
        const matrix_double& xt = pre_layer->_data[t];
        
        matrix_double f_output;
        if (! _use_peephole) {
            f_output = sigmoid_m(xt * _fg_input_weights 
                + pre_hidden_vals * _fg_hidden_weights + _fg_bias);
        } else {
            f_output = sigmoid_m(xt * _fg_input_weights
                + pre_hidden_vals * _fg_hidden_weights 
                + pre_cell_vals * _fg_cell_weights + _fg_bias);
        }
    
        matrix_double i_output;
        if (! _use_peephole) {
            i_output = sigmoid_m(xt * _ig_input_weights 
                + pre_hidden_vals * _ig_hidden_weights + _ig_bias);
        } else {
            i_output = sigmoid_m(xt * _ig_input_weights 
                + pre_hidden_vals * _ig_hidden_weights 
                + pre_cell_vals * _ig_cell_weights
                + _ig_bias);
        }
    
        matrix_double o_output;
        if (! _use_peephole) {
            o_output = sigmoid_m(xt * _og_input_weights 
                + pre_hidden_vals * _og_hidden_weights + _og_bias);
        } else {
            o_output = sigmoid_m(xt * _og_input_weights 
                + pre_hidden_vals * _og_hidden_weights
                + pre_cell_vals * _og_cell_weights + _og_bias);
        }
    
        matrix_double cell_new_val = tanh_m(xt * _cell_input_weights 
            + pre_hidden_vals * _cell_hidden_weights + _cell_bias);
    
        matrix_double cell_output = cell_new_val.dot_mul(i_output) + pre_cell_vals.dot_mul(f_output);
        pre_cell_vals = cell_output;
        pre_hidden_vals = tanh_m(cell_output).dot_mul(o_output);
        //
        _lstm_layer_values._cell_values.set_row(t, cell_output);
        _lstm_layer_values._hidden_values.set_row(t, pre_hidden_vals);
        _lstm_layer_values._og_values.set_row(t, o_output);
        _lstm_layer_values._ig_values.set_row(t, i_output);
        _lstm_layer_values._fg_values.set_row(t, f_output);
        _lstm_layer_values._cell_new_values.set_row(t, cell_new_val);
        _data.push_back(pre_hidden_vals);
    }
    _pre_layer = pre_layer;
    _pre_layer_data = pre_layer->_data;
}

void LstmCell::_backward(Layer* nxt_layer) {
    if (nxt_layer->_type != SEQ_FULL 
        && nxt_layer->_type != RNN_CELL
        && nxt_layer->_type != LSTM_CELL) {
        exit(1);
    }
    std::vector<matrix_double>().swap(_fg_errors);
    std::vector<matrix_double>().swap(_ig_errors);
    std::vector<matrix_double>().swap(_og_errors);
    std::vector<matrix_double>().swap(_new_cell_errors);
    
    _ig_delta_input_weights.resize(_input_dim, _output_dim);
    _ig_delta_hidden_weights.resize(_output_dim, _output_dim);
    _ig_delta_cell_weights.resize(_output_dim, _output_dim);
    _ig_delta_bias.resize(1, _output_dim);
    
    _fg_delta_input_weights.resize(_input_dim, _output_dim);
    _fg_delta_hidden_weights.resize(_output_dim, _output_dim);
    _fg_delta_cell_weights.resize(_output_dim, _output_dim);
    _fg_delta_bias.resize(1, _output_dim);
    
    _og_delta_input_weights.resize(_input_dim, _output_dim);
    _og_delta_hidden_weights.resize(_output_dim, _output_dim);
    _og_delta_cell_weights.resize(_output_dim, _output_dim);
    _og_delta_bias.resize(1, _output_dim);
    
    _cell_delta_input_weights.resize(_input_dim, _output_dim);
    _cell_delta_hidden_weights.resize(_output_dim, _output_dim);
    _cell_delta_bias.resize(1, _output_dim);

    matrix_double fg_error(1, _output_dim);
    matrix_double ig_error(1, _output_dim);
    matrix_double og_error(1, _output_dim);
    matrix_double new_cell_error(1, _output_dim);

    matrix_double nxt_fg_error(1, _output_dim);
    matrix_double nxt_ig_error(1, _output_dim);
    matrix_double nxt_og_error(1, _output_dim);
    matrix_double nxt_new_cell_error(1, _output_dim);
    matrix_double nxt_cell_mid_error(1, _output_dim);

    // the layer after the lstm cell is not only seq_full_conn_layer, so 
    // the weight & error are all needed to be calculated
    std::vector<matrix_double> nxt_layer_error_weights;
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

        matrix_double hidden_mid_error = nxt_layer_error_weights[t] 
            + nxt_fg_error * _fg_hidden_weights._T() \
            + nxt_ig_error * _ig_hidden_weights._T() \
            + nxt_og_error * _og_hidden_weights._T() \
            + nxt_new_cell_error * _cell_hidden_weights._T();

        matrix_double cell_mid_error = hidden_mid_error
            .dot_mul(_lstm_layer_values._og_values._R(t))
            .dot_mul(tanh_m_diff(tanh_m(_lstm_layer_values._cell_values._R(t))));
        
        if (t + 1 < _seq_len) {
            cell_mid_error = cell_mid_error 
                + nxt_cell_mid_error.dot_mul(_lstm_layer_values._fg_values._R(t + 1));
        }
        if (_use_peephole) {
            cell_mid_error = cell_mid_error + nxt_fg_error * _fg_cell_weights._T() 
                + nxt_ig_error * _ig_cell_weights._T()
                + nxt_og_error * _og_cell_weights._T();
        }
        // output gate error
        og_error = hidden_mid_error
            .dot_mul(tanh_m(_lstm_layer_values._cell_values._R(t)))
            .dot_mul(sigmoid_m_diff(_lstm_layer_values._og_values._R(t)));
        // input gate error
        ig_error = cell_mid_error
            .dot_mul(_lstm_layer_values._cell_new_values._R(t))
            .dot_mul(sigmoid_m_diff(_lstm_layer_values._ig_values._R(t)));
        // forget gate error
        fg_error.resize(0.0);
        if (t > 0) {
            fg_error = cell_mid_error
                .dot_mul(_lstm_layer_values._cell_values._R(t - 1))
                .dot_mul(sigmoid_m_diff(_lstm_layer_values._fg_values._R(t)));
        }

        // new cell error
        new_cell_error = cell_mid_error
            .dot_mul(_lstm_layer_values._ig_values._R(t))
            .dot_mul(tanh_m_diff(_lstm_layer_values._cell_new_values._R(t)));
        // add delta

        _fg_errors.push_back(fg_error);
        _ig_errors.push_back(ig_error);
        _og_errors.push_back(og_error);
        _new_cell_errors.push_back(new_cell_error);

        const matrix_double& xt_trans = _pre_layer_data[t]._T();
        
        _ig_delta_input_weights.add(xt_trans * ig_error);
        _fg_delta_input_weights.add(xt_trans * fg_error);
        _og_delta_input_weights.add(xt_trans * og_error);
        _cell_delta_input_weights.add(xt_trans * new_cell_error);

        if (t > 0) {
            const matrix_double& hidden_value_pre = _lstm_layer_values._hidden_values._R(t - 1)._T();
            _ig_delta_hidden_weights.add(hidden_value_pre * ig_error);
            _og_delta_hidden_weights.add(hidden_value_pre * og_error);
            _fg_delta_hidden_weights.add(hidden_value_pre * fg_error);
            _cell_delta_hidden_weights.add(hidden_value_pre * new_cell_error);
            
            if (_use_peephole) {
                const matrix_double& cell_value_pre = _lstm_layer_values._cell_values._R(t - 1)._T();
                _ig_delta_cell_weights.add(cell_value_pre * ig_error);
                _og_delta_cell_weights.add(cell_value_pre * og_error);
                _fg_delta_cell_weights.add(cell_value_pre * fg_error);
            }
        }
        
        _ig_delta_bias.add(ig_error);
        _og_delta_bias.add(og_error);
        _fg_delta_bias.add(fg_error);
        _cell_delta_bias.add(new_cell_error);
        nxt_ig_error = ig_error;
        nxt_og_error = og_error;
        nxt_fg_error = fg_error;
        nxt_new_cell_error = new_cell_error;
        nxt_cell_mid_error = cell_mid_error;

    }
    std::reverse(_fg_errors.begin(), _fg_errors.end());
    std::reverse(_ig_errors.begin(), _ig_errors.end());
    std::reverse(_og_errors.begin(), _og_errors.end());
    std::reverse(_new_cell_errors.begin(), _new_cell_errors.end());
}

void LstmCell::_update_gradient(int opt_type, double learning_rate) {
    if (opt_type == SGD) {
        _ig_input_weights.add(_ig_delta_input_weights * learning_rate);
        _ig_hidden_weights.add(_ig_delta_hidden_weights * learning_rate);
        _ig_bias.add(_ig_delta_bias * learning_rate);
        
        _fg_input_weights.add(_fg_delta_input_weights * learning_rate);
        _fg_hidden_weights.add(_fg_delta_hidden_weights * learning_rate);
        _fg_bias.add(_fg_delta_bias * learning_rate);
        
        _og_input_weights.add(_og_delta_input_weights * learning_rate);
        _og_hidden_weights.add(_og_delta_hidden_weights * learning_rate);
        _og_bias.add(_og_delta_bias * learning_rate);
        
        _cell_input_weights.add(_cell_delta_input_weights * learning_rate);
        _cell_hidden_weights.add(_cell_delta_hidden_weights * learning_rate);
        _cell_bias.add(_cell_delta_bias * learning_rate);

        if (_use_peephole) {
            _ig_cell_weights.add(_ig_delta_cell_weights * learning_rate);
            _fg_cell_weights.add(_fg_delta_cell_weights * learning_rate);
            _og_cell_weights.add(_og_delta_cell_weights * learning_rate);
        }
        
    }
}

}
