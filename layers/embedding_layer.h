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

namespace sub_dl {

class SeqEmbeddingLayer : public Layer {

public:
    matrix_double _embedding_vec;
    matrix_double _delta_embedding_vec;
    int _voca_size;
    float _learning_rate;

public:
    SeqEmbeddingLayer(int voca_size, int output_dim) {
        
        _type = EMB;
        _voca_size = voca_size;
        _output_dim = output_dim;
        _embedding_vec.resize(voca_size, output_dim);
        _embedding_vec.assign_val();
        _delta_embedding_vec.resize(voca_size, output_dim);
    }

    ~SeqEmbeddingLayer() {
    }

    void _set_learning_rate(float learning_rate) {
        _learning_rate = learning_rate;
    }

    void _forward(Layer* pre_layer) {
        std::vector<matrix_double>().swap(_data);
        if (pre_layer->_type != INPUT) {
            exit(1);
        }
        _seq_len = pre_layer->_data.size();
        for (int i = 0; i < _seq_len; i++) {
            int word_id = pre_layer->_data[i][0][0];
            if (word_id > _embedding_vec._x_dim) {
                exit(1);
            }
            // matrix_double feature(1, _output_dim);
            _data.push_back(_embedding_vec._R(word_id - 1));
        }
        _pre_layer = pre_layer;
    }

    void _backward(Layer* nxt_layer) {

        if (nxt_layer->_type != RNN_CELL &&
            nxt_layer->_type != LSTM_CELL &&
            nxt_layer->_type != GRU_CELL) {
            exit(1);
        }
        std::vector<matrix_double> nxt_layer_error_weights; 
        if (nxt_layer->_type == RNN_CELL) {
            RnnCell* rnn_cell = (RnnCell*) nxt_layer;
            for (int t = 0; t < _seq_len; t++) {
                nxt_layer_error_weights.push_back(rnn_cell->_errors[t] * rnn_cell->_input_hidden_weights._T());
            }
        } else if (nxt_layer->_type == GRU_CELL) {
            GruCell* gru_cell = (GruCell*) nxt_layer;
            for (int t = 0; t < _seq_len; t++) {
                nxt_layer_error_weights.push_back(gru_cell->_ug_errors[t] * gru_cell->_ug_input_weights._T()
                + gru_cell->_rg_errors[t] * gru_cell->_rg_input_weights._T()
                + gru_cell->_newh_errors[t]  * gru_cell->_newh_input_weights._T());
            }
        } else if (nxt_layer->_type == LSTM_CELL) {
			LstmCell* lstm_cell = (LstmCell*) nxt_layer;
			for (int i = 0; i < _seq_len; i++) {
				nxt_layer_error_weights.push_back(lstm_cell->_fg_errors[i] * lstm_cell->_fg_input_weights._T()
					+ lstm_cell->_ig_errors[i] * lstm_cell->_ig_input_weights._T()
					+ lstm_cell->_og_errors[i] * lstm_cell->_og_input_weights._T()
					+ lstm_cell->_new_cell_errors[i] * lstm_cell->_cell_input_weights._T());
			}
		}

        int wid = 0;
        for (int t = _seq_len - 1; t >= 0; t--) {
            int word_id = _pre_layer->_data[t][0][0];
            matrix_double old_embedding = _delta_embedding_vec._R(word_id - 1);
            old_embedding.add(nxt_layer_error_weights[t]);
            _delta_embedding_vec.set_row(word_id - 1, old_embedding);
        }
    }

    void _clear_gradient() {
        std::cout << "clear graident!" << std::endl;
        _delta_embedding_vec.resize(0.0);
    }
    
    void display() {
    }

    
    void _update_gradient(int opt_type, double learning_rate) {
        for (int t = _seq_len - 1; t >= 0; t--) {
            int word_id = _pre_layer->_data[t][0][0];
            matrix_double old_embedding = _embedding_vec._R(word_id - 1);
            old_embedding.add(_delta_embedding_vec._R(word_id - 1) * _learning_rate);
            _embedding_vec.set_row(word_id - 1, old_embedding);
        }
    }
};

class WordEmbeddingLayer : public Layer {

public:

    WordEmbeddingLayer(int output_dim) {
        _output_dim = output_dim;
        _type = EMB;
    }
    
    /*
    * @brief embeeding layer is only available for binary decoding
    *
    */
    void _forward(Layer* pre_layer) {
        std::vector<matrix_double>().swap(_data);
        if (pre_layer->_type != INPUT) {
            exit(1);
        }
        _seq_len = pre_layer->_data.size();
        for (int i = 0; i < _seq_len; i++) {
            int word_id = pre_layer->_data[i][0][0];
            int idx = 0;
            matrix_double feature(1, _output_dim);
            while (word_id > 0) {
                feature[0][idx ++] = word_id % 2;
                word_id /= 2;
            }
            for (; idx < _output_dim; idx++) {
                feature[0][idx] = 0;
            }
            _data.push_back(feature);
        }
    }
    void _backward(Layer* nxt_layer) {}
    void display() {}
    void _update_gradient(int opt_type, double learning_rate) {}
    void _clear_gradient() {}
};

}
