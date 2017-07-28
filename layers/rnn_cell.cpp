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
#include "rnn_cell.h"
#include "lstm_cell.h"
#include "gru_cell.h"
#include "seq_full_conn_layer.h"

namespace sub_dl {

RnnCell::RnnCell(int input_dim, int output_dim) {
	
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

void RnnCell::_forward(Layer* pre_layer) {
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

void RnnCell::_backward(Layer* nxt_layer) {
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
		matrix_double hidden_error = (nxt_layer_error_weights[t]
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

void RnnCell::_update_gradient(int opt_type, double learning_rate) {
	if (opt_type == SGD) {
		_input_hidden_weights.add(_delta_input_hidden_weights * learning_rate);
		_hidden_bias.add(_delta_hidden_bias * learning_rate);
		_hidden_weights.add(_delta_hidden_weights * learning_rate);
	}
}

}

