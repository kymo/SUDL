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

#ifndef BI_CELL_WRAPPER_H
#define BI_CELL_WRAPPER_H

#include "layer.h"
#include "rnn_cell.h"
#include "lstm_cell.h"
#include "gru_cell.h"

namespace sub_dl {

// the T stands for the different type of rnn cells, T can be RnnCell 
// LstmCell or GruCell if you want to build Bi-directional cell version
template <typename T>
class BiCellWrapper : public Layer {

public:
    
    // positive sequence cell, from t = 1 to T
    T *_pos_seq_cell;
    // negative sequence cell, from t = T to 1
    T *_neg_seq_cell;
    
    /*
    * @brief constuct function for bi-lstm
    *
    * @param
    *    seq_input_dim: feature dimention in time t
    *    seq_output_dim: label dimention in time t
    *    pos_use_peephole: peephole switch for positive order lstm cell
    *    neg_use_peephole: peephole switch for negative order lstm cell
    *
    * @return
    *    void
    * 
    */
    BiCellWrapper(int seq_input_dim, int seq_output_dim,
        bool pos_use_peephole,
        bool neg_use_peephole,
        int cell_type) {

        _pos_seq_cell = new T(seq_input_dim, seq_output_dim, pos_use_peephole);
        _neg_seq_cell = new T(seq_input_dim, seq_output_dim, neg_use_peephole);
        _type = cell_type;
    
    }

    /*
    * @param
    *    seq_input_dim: feature dimention in time t
    *    seq_output_dim: label dimention in time t
    *    cell_type:
    *        BI_GRU_CELL GruCell
    *        BI_LSTM_CELL LstmCell
    *        BI_GRU_CELL GruCell
    *
    * @return
    *    void
    * 
    */
    BiCellWrapper(int seq_input_dim,
        int seq_output_dim,
        int cell_type) {
    
        _pos_seq_cell = new T(seq_input_dim, seq_output_dim);
        _neg_seq_cell = new T(seq_input_dim, seq_output_dim);
        _type = cell_type;
    }

    ~BiCellWrapper() {
        if (_pos_seq_cell != NULL) {
            delete _pos_seq_cell;
            _pos_seq_cell = NULL;
        }
        if (_neg_seq_cell != NULL) {
            delete _neg_seq_cell;
            _neg_seq_cell = NULL;
        }
    }

    /*
    * @brief forward process of bi-directional cell, if the pre layer of 
    *     of current cell if also bi-directional cell, just link the input
    *    and output of each time.
    *
    * @param
    *    pre_layer: pre layer
    *
    * @return
    *    void
    */
    void _forward(Layer* pre_layer) {
        std::vector<matrix_double>().swap(_data);
        _seq_len = pre_layer->_seq_len;
        if (pre_layer->_type == BI_RNN_CELL) {
            BiCellWrapper<RnnCell>* bi_rnn_cell = (BiCellWrapper<RnnCell>*) pre_layer;
            _pos_seq_cell->_forward(bi_rnn_cell->_pos_seq_cell);
            _neg_seq_cell->_forward(bi_rnn_cell->_neg_seq_cell);
        } else if (pre_layer->_type == BI_LSTM_CELL) {
            BiCellWrapper<LstmCell>* bi_lstm_cell = (BiCellWrapper<LstmCell>*) pre_layer;
            _pos_seq_cell->_forward(bi_lstm_cell->_pos_seq_cell);
            _neg_seq_cell->_forward(bi_lstm_cell->_neg_seq_cell);
        } else if (pre_layer->_type == BI_GRU_CELL) {
            BiCellWrapper<GruCell>* bi_gru_cell = (BiCellWrapper<GruCell>*) pre_layer;
            _pos_seq_cell->_forward(bi_gru_cell->_pos_seq_cell);
            _neg_seq_cell->_forward(bi_gru_cell->_neg_seq_cell);
        } else {
            _pos_seq_cell->_forward(pre_layer);
            // reverse the pre layer data
            std::reverse(pre_layer->_data.begin(), pre_layer->_data.end());
            _neg_seq_cell->_forward(pre_layer);
            // reverse back for backward
            std::reverse(pre_layer->_data.begin(), pre_layer->_data.end());
        }
        for (int t = 0; t < _seq_len; t++) {
            _data.push_back(_pos_seq_cell->_data[t] + 
                _neg_seq_cell->_data[_seq_len - 1 - t]);
        }
        _pre_layer = pre_layer;
    }

    void _backward(Layer* nxt_layer) {
        
        if (nxt_layer->_type == BI_RNN_CELL) {
            BiCellWrapper<RnnCell>* bi_rnn_cell = (BiCellWrapper<RnnCell>*) nxt_layer;
            _pos_seq_cell->_backward(bi_rnn_cell->_pos_seq_cell);
            _neg_seq_cell->_backward(bi_rnn_cell->_neg_seq_cell);
        } else if (nxt_layer->_type == BI_LSTM_CELL) {
            BiCellWrapper<LstmCell>* bi_lstm_cell = (BiCellWrapper<LstmCell>*) nxt_layer;
            _pos_seq_cell->_backward(bi_lstm_cell->_pos_seq_cell);
            _neg_seq_cell->_backward(bi_lstm_cell->_neg_seq_cell);
        } else if (nxt_layer->_type == BI_GRU_CELL) {
            BiCellWrapper<GruCell>* bi_gru_cell = (BiCellWrapper<GruCell>*) nxt_layer;
            _pos_seq_cell->_backward(bi_gru_cell->_pos_seq_cell);
            _neg_seq_cell->_backward(bi_gru_cell->_neg_seq_cell);
        } else {
            _pos_seq_cell->_backward(nxt_layer);
            // reverse the pre layer error
            std::reverse(nxt_layer->_errors.begin(), nxt_layer->_errors.end());
            // reverse the pre layer error
            _neg_seq_cell->_backward(nxt_layer);
            std::reverse(nxt_layer->_errors.begin(), nxt_layer->_errors.end());
        }
    }

    void display() {}

    void _update_gradient(int opt_type, double learning_rate) {
        if (opt_type == SGD) {
            _pos_seq_cell->_update_gradient(opt_type, learning_rate);
            _neg_seq_cell->_update_gradient(opt_type, learning_rate);
        }
    }
};

}

#endif
