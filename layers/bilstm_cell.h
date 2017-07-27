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

#ifndef BI_LSTM_CELL_H
#define BI_LSTM_CELL_H

#include "layer.h"
#include "lstm_cell.h"

namespace sub_dl {

class BiLstmCell : public Layer {

public:
    LstmCell *_pos_seq_lstm_cell;
    LstmCell *_neg_seq_lstm_cell;

    int _pos_seq_input_dim;
    int _pos_seq_output_dim;
    int _neg_seq_input_dim;
    int _neg_seq_output_dim;

    bool _pos_use_peephole;
    bool _neg_use_peephole;

    BiLstmCell(int pos_seq_input_dim,
        int pos_seq_output_dim,
        bool pos_use_peephole,
        int neg_seq_input_dim,
        int neg_seq_output_dim,
        bool neg_use_peephole) {

        _pos_seq_input_dim = pos_seq_input_dim;
        _pos_seq_output_dim = pos_seq_output_dim;
        _pos_use_peephole = pos_use_peephole;    
        _neg_seq_input_dim = neg_seq_input_dim;
        _neg_seq_output_dim = neg_seq_output_dim;
        _neg_use_peephole = neg_use_peephole;

        _pos_seq_lstm_cell = new LstmCell(pos_seq_input_dim,
            pos_seq_output_dim, pos_use_peephole);
        _neg_seq_lstm_cell = new LstmCell(neg_seq_input_dim,
            neg_seq_output_dim, neg_use_peephole);

        _type = BI_LSTM_CELL;

    }

    void _forward(Layer* pre_layer) {

        if (pre_layer->_type == BI_LSTM_CELL) {

            BiLstmCell* bi_lstm_cell = (BiLstmCell*) pre_layer;
            _pos_seq_lstm_cell->_forward(bi_lstm_cell->_pos_seq_lstm_cell);
            _neg_seq_lstm_cell->_forward(bi_lstm_cell->_neg_seq_lstm_cell);

        } else {
            _seq_len = pre_layer->_seq_len;
            _pos_seq_lstm_cell->_forward(pre_layer);
            // reverse the pre layer data
            std::reverse(pre_layer->_data.begin(), pre_layer->_data.end());
            _neg_seq_lstm_cell->_forward(pre_layer);
            // reverse back for backward
            std::reverse(pre_layer->_data.begin(), pre_layer->_data.end());
            for (int t = 0; t < _seq_len; t++) {
                _data.push_back(_pos_seq_lstm_cell->_data[t] + 
                    _neg_seq_lstm_cell->_data[_seq_len - 1 - t]);
            }
        }
        _pre_layer = pre_layer;
    }

    void _backward(Layer* nxt_layer) {
        
        if (nxt_layer->_type == BI_LSTM_CELL) {
            BiLstmCell* bi_lstm_cell = (BiLstmCell*) nxt_layer;
            _pos_seq_lstm_cell->_backward(bi_lstm_cell->_pos_seq_lstm_cell);
            _neg_seq_lstm_cell->_backward(bi_lstm_cell->_neg_seq_lstm_cell);
        } else {
            _pos_seq_lstm_cell->_backward(nxt_layer);
            std::reverse(nxt_layer->_errors.begin(), nxt_layer->_errors.end());
            _neg_seq_lstm_cell->_backward(nxt_layer);
            std::reverse(nxt_layer->_errors.begin(), nxt_layer->_errors.end());
        }
    }

    void display() {
    }

    void _update_gradient(int opt_type, double learning_rate) {
        if (opt_type == SGD) {
            _pos_seq_lstm_cell->_update_gradient(opt_type, learning_rate);
            _neg_seq_lstm_cell->_update_gradient(opt_type, learning_rate);
        }
    }
};

};

#endif
