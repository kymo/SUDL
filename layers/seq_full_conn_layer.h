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

#ifndef SEQ_FULL_CONN_LAYER_
#define SEQ_FULL_CONN_LAYER_

#include "layer.h"

namespace sub_dl {

class SeqFullConnLayer : public Layer {

public:

    matrix_double _seq_full_weights;
    matrix_double _seq_full_bias;
    matrix_double _delta_seq_full_weights;
    matrix_double _delta_seq_full_bias;
    
    SeqFullConnLayer(int input_dim, int output_dim) {
        _type = SEQ_FULL;
        _input_dim = input_dim;
        _output_dim = output_dim;

        _seq_full_weights.resize(_input_dim, _output_dim);
        _seq_full_bias.resize(1, _output_dim);

        _seq_full_weights.assign_val();
        _seq_full_bias.assign_val();
        
        _delta_seq_full_weights.resize(_input_dim, _output_dim);
        _delta_seq_full_bias.resize(1, _output_dim);

    }

    void _forward(Layer* pre_layer) {
        std::vector<matrix_double>().swap(_data);

        if (pre_layer->_type != RNN_CELL 
            && pre_layer->_type != LSTM_CELL
            && pre_layer->_type != GRU_CELL
            && pre_layer->_type != BI_LSTM_CELL
            && pre_layer->_type != BI_GRU_CELL
            && pre_layer->_type != BI_RNN_CELL) { 
            FATAL_LOG("Layer before seq faull conn layer is not rnn cell! func[%s] line[%d]", __func__, __LINE__);
            exit(1);
        }
        _seq_len = pre_layer->_seq_len;

        for (int i = 0; i < _seq_len; i++) {
            matrix_double value = pre_layer->_data[i] 
                * _seq_full_weights + _seq_full_bias;
            _data.push_back(value);
        }
        _pre_layer = pre_layer;

    }

    void _backward(Layer* nxt_layer) {
        
        if (nxt_layer->_type != SEQ_ACT) {
            FATAL_LOG("next layer is not seq loss layer! func[%s] line[%d]", __func__, __LINE__);
            exit(1);
        }
        std::vector<matrix_double>().swap(_errors);
        _errors = nxt_layer->_errors;
        for (int i = _seq_len - 1; i >= 0; i--) {
            _delta_seq_full_weights.add(_pre_layer->_data[i]._T() * _errors[i]);
            _delta_seq_full_bias.add(_errors[i]);
        }

    }

    void display() {}

    void _update_gradient(int opt_type, double learning_rate) {
        if (opt_type == SGD) {
            _seq_full_weights.add(_delta_seq_full_weights * learning_rate);
            _seq_full_bias.add(_delta_seq_full_bias * learning_rate);
        }
    }

    void _clear_gradient() {
        _delta_seq_full_weights.resize(0.0);
        _delta_seq_full_bias.resize(0.0);
    }
};

}
#endif
