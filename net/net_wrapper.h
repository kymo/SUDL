// Copyright (c) 2017. kymowind@gmail.com. All Rights Reserve.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//    http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License. 

#ifndef NET_WRAPPER_H
#define NET_WRAPPER_H

#include <iostream>
#include "seq_full_conn_layer.h"
#include "seq_full_conn_softmax_layer.h"
#include "seq_loss_layer.h"
#include "rnn_cell.h"
#include "lstm_cell.h"
#include "gru_cell.h"
#include "bi_cell_wrapper.h"
#include "matrix.h"
#include "embedding_layer.h"
#include "conv_layer.h"
#include "pooling_layer.h"
#include "full_conn_layer.h"
#include "full_conn_softmax_layer.h"
#include "active_layer.h"
#include "loss_layer.h"
#include "util.h"

#include <fstream>

#define EMBEDDING_DIM 14
#define LABEL_DIM 4
#define SAMPLE_SEP ";"
#define FEATURE_SEP " "
#define LABEL_SEP " "
#define SAMPLE_SEP_SIZE 2


namespace sub_dl {

// net wrapper is used to build a rnn or cnn model or common 
// bp neural netowrk, T indicates the type of neural netowrk
template <typename T>
class NetWrapper {

public:
    // input dim
    int _input_dim;
    // output dim
    int _output_dim;

    NetWrapper(int output_dim) {
        _output_dim = output_dim;
        _data_layer = new DataFeedLayer();
        _loss_layer = new T();
    }

    ~NetWrapper() {
        if (_data_layer != NULL) {
            delete _data_layer;
            _data_layer = NULL;
        }
        if (_loss_layer != NULL) {
            delete _loss_layer;
            _loss_layer = NULL;
        }
    }

    // data layer
    DataFeedLayer* _data_layer;

    // loss layer
    T* _loss_layer;
    
    // layers of the net
    std::vector<Layer*> _layers;

    /*
    * @brief build network
    *
    * @param
    *    layers: different layers of model
    *
    * @return
    *    voie
    */
    void _build_net(const std::vector<Layer*> layers) {
        _layers = layers;
    }

    /* 
    * @brief 
    *    calculate output of each layer of rnn 
    *
    * @param
    *    feature: the current feature of input sample
    *
    * @return
    */
    void _forward(const std::vector<matrix_double>& feature) {
        // const matrix_double& feature) {
        _data_layer->_set_data(feature);
        Layer* pre_layer = _data_layer;
        for (auto layer : _layers) {
            layer->_forward(pre_layer);
            pre_layer = layer;
        }
    }
    
    /*
    * @brief
    *    update weigths of each layers
    *
    * @return
    *    void
    */
    void _update_gradient() {
        for (auto layer : _layers) {
            layer->_update_gradient(SGD, -0.001);
        }
    }
    /*
    * @brief
    *    clear data cache & errro cache
    *
    * @return
    *    void
    *
    */
    void _clear_gradient() {
        for (auto layer : _layers) {
            layer->_clear_gradient();
        }
    }

    void _gradient_weights_check(const std::vector<std::vector<matrix_double> >& batch_x,
        const std::vector<matrix_double>& batch_y,
        matrix_double& weights,
        const matrix_double& delta_weights) {
        for (int i = 0; i < weights._x_dim; i++) {
            for (int j = 0; j < weights._y_dim; j++) {
                
                double v = weights[i][j];
                weights[i][j] = v + 1.0e-4;
                double f1 = 0.0;
                double f2 = 0.0;
                   T* nxt_layer;
                for (int k = 0; k < batch_x.size(); k++) {    
                    matrix_double label;
                    label_encode(batch_y[k], label, _output_dim);
                    _forward(batch_x[k]);
                    nxt_layer = new T();
                    nxt_layer->_set_label(label);
                    nxt_layer->_forward(_layers.back());
                    nxt_layer->_backward(NULL);
                    for (int k = 0; k < nxt_layer->_data.size(); k++) {
                        f1 += nxt_layer->_data[k].sum();
                    }
                    delete nxt_layer;
                }

                weights[i][j] = v - 1.0e-4;
                for (int k = 0; k < batch_x.size(); k++) {
                   
                    matrix_double label;
                    label_encode(batch_y[k], label, _output_dim);
                    _forward(batch_x[k]);
                    nxt_layer = new T();
                    nxt_layer->_set_label(label);
                    nxt_layer->_forward(_layers.back());
                    nxt_layer->_backward(NULL);
                    for (int k = 0; k < nxt_layer->_data.size(); k++) {
                        f2 += nxt_layer->_data[k].sum();
                    }
                    delete nxt_layer;
                }
                std::cout << "[ " << delta_weights[i][j] << "," << (f1 - f2) / (2.0e-4) << "]";
                
                weights[i][j] = v;
            }
        }
    }
    
    double _gradient_check(const std::vector<std::vector<matrix_double> >& batch_x,
        const std::vector<matrix_double>& batch_y) {
        
        for (int l = 0; l < _layers.size(); l++) {
            if (_layers[l]->_type == SEQ_FULL) {
                std::cout << "------------Gradient Check for seq full layer -------------" << std::endl;
                SeqFullConnLayer* seq_full = (SeqFullConnLayer*) _layers[l];
                _gradient_weights_check(batch_x, batch_y, 
                    seq_full->_seq_full_weights, seq_full->_delta_seq_full_weights);
                _gradient_weights_check(batch_x, batch_y, 
                    seq_full->_seq_full_bias, seq_full->_delta_seq_full_bias);
            } else if (_layers[l]->_type == RNN_CELL) {
                std::cout << "------------Gradient Check for rnn cell layer -------------" << std::endl;
                RnnCell* rnn_cell = (RnnCell*) _layers[l];
                _gradient_weights_check(batch_x, batch_y, 
                    rnn_cell->_input_hidden_weights, rnn_cell->_delta_input_hidden_weights); 
            } else if (_layers[l]->_type == LSTM_CELL) {
                std::cout << "------------Gradient Check for lstm cell layer -------------" << std::endl;
                LstmCell* lstm_cell = (LstmCell*) _layers[l];
                _gradient_weights_check(batch_x, batch_y, 
                    lstm_cell->_ig_input_weights, lstm_cell->_ig_delta_input_weights);
            } else if (_layers[l]->_type == BI_LSTM_CELL) {
                std::cout << "------------Gradient Check for bi lstm cell layer -------------" << std::endl;
                LstmCell* lstm_cell = ((BiCellWrapper<LstmCell>*) _layers[l])->_pos_seq_cell;
                _gradient_weights_check(batch_x, batch_y, 
                    lstm_cell->_ig_input_weights, lstm_cell->_ig_delta_input_weights);
                _gradient_weights_check(batch_x, batch_y, lstm_cell->_og_input_weights, lstm_cell->_og_delta_input_weights);
                _gradient_weights_check(batch_x, batch_y, lstm_cell->_fg_input_weights, lstm_cell->_fg_delta_input_weights);
                _gradient_weights_check(batch_x, batch_y, lstm_cell->_cell_input_weights, lstm_cell->_cell_delta_input_weights);
            } else if (_layers[l]->_type == BI_RNN_CELL) {
                std::cout << "------------Gradient Check for bi rnn cell layer -------------" << std::endl;
                RnnCell* rnn_cell = ((BiCellWrapper<RnnCell>*) _layers[l])->_pos_seq_cell;
                _gradient_weights_check(batch_x, batch_y, rnn_cell->_input_hidden_weights, rnn_cell->_delta_input_hidden_weights);
                _gradient_weights_check(batch_x, batch_y, rnn_cell->_hidden_weights, rnn_cell->_delta_hidden_weights);
            } else if (_layers[l]->_type == BI_GRU_CELL) {
                std::cout << "------------Gradient Check for bi gru cell layer -------------" << std::endl;
                GruCell* gru_cell = ((BiCellWrapper<GruCell>*) _layers[l])->_pos_seq_cell;
                _gradient_weights_check(batch_x, batch_y, gru_cell->_ug_hidden_weights, gru_cell->_delta_ug_hidden_weights); 
                _gradient_weights_check(batch_x, batch_y, gru_cell->_rg_hidden_weights, gru_cell->_delta_rg_hidden_weights); 
                _gradient_weights_check(batch_x, batch_y, gru_cell->_newh_hidden_weights, gru_cell->_delta_newh_hidden_weights); 
                _gradient_weights_check(batch_x, batch_y, gru_cell->_newh_input_weights, gru_cell->_delta_newh_input_weights); 

            } else if (_layers[l]->_type == GRU_CELL) {
                std::cout << "------------Gradient Check for gru cell layer -------------" << std::endl;
                GruCell* gru_cell = (GruCell*) _layers[l];
                _gradient_weights_check(batch_x, batch_y, gru_cell->_ug_hidden_weights, gru_cell->_delta_ug_hidden_weights); 
                _gradient_weights_check(batch_x, batch_y, gru_cell->_rg_hidden_weights, gru_cell->_delta_rg_hidden_weights); 
                _gradient_weights_check(batch_x, batch_y, gru_cell->_newh_hidden_weights, gru_cell->_delta_newh_hidden_weights); 
                _gradient_weights_check(batch_x, batch_y, gru_cell->_newh_input_weights, gru_cell->_delta_newh_input_weights); 
            } else if (_layers[l]->_type == CONV) {
                ConvLayer* layer = (ConvLayer*) _layers[l];
                std::cout << "-------Conv Layer Gradient Check Result ---------" << std::endl;
                for (int i = 0; i < layer->_input_dim;i ++) {
                    for (int j = 0; j < layer->_output_dim; j++) {
                        if (layer->_conn_map[i][j]) {
                            _gradient_weights_check(batch_x, batch_y, 
                                layer->_conv_kernels[i][j], layer->_delta_conv_kernels[i][j]);
                         }
                    }
                }
            } else if (_layers[l]->_type == FULL_CONN) {
                BaseFullConnLayer* layer = (BaseFullConnLayer*) _layers[l];
                std::cout << "-------Full Layer Gradient Check Result ---------" << std::endl;
                _gradient_weights_check(batch_x, batch_y, layer->_full_conn_weights, layer->_delta_full_conn_weights);

            }
        }
    }

    double _backward(const matrix_double& label) {
        if (_loss_layer == NULL) {
            FATAL_LOG("Error when call loss layer! func[%s] line[%d]", __func__, __LINE__);
            exit(1);
        }
        _loss_layer->_set_label(label);
        _loss_layer->_forward(_layers.back());
        _loss_layer->_backward(NULL);
        Layer* nxt_layer = _loss_layer;
        double cost = 0.0;
        for (int i = 0; i < _loss_layer->_data.size(); i++) {
            cost += _loss_layer->_data[i].sum();
        }
        for (int j = _layers.size() - 1; j >= 0; j--) {
            _layers[j]->_backward(nxt_layer);
            nxt_layer = _layers[j];
        }

        return cost;
    }

    /*
    * @brief rnn train process,because the input data of rnn and cnn are not
    *    the same dimention, so the training process should be splited 
    * 
    * @param
    *    batch_x: batch sample feature maps
    *    batch_y: batch sample labels
    *
    * @return
    *
    */
    double _train(const std::vector<std::vector<matrix_double> >& batch_x,
        const std::vector<matrix_double>& batch_y) {
        
        double cost = 0.0;
        std::string val1, val2;
        _clear_gradient();
        for (int i = 0; i < batch_x.size(); i++) {
            matrix_double label;
            label_encode(batch_y[i], label, _output_dim);
            // forward
            _forward(batch_x[i]);
            cost += _backward(label);
        }
        // _gradient_check(batch_x, batch_y);
        _update_gradient();
        return cost;
    }

    void _predict(const std::vector<matrix_double>& feature, std::vector<int>& labels) {
        
        _forward(feature);
        const Layer* layer = _layers.back();
        for (int i = 0; i < layer->_data.size(); i++) {
            int mx_id = 0;
            float val = -100;
            for (int j = 0; j < layer->_data[i]._y_dim; j++) {
                if (val < layer->_data[i][0][j]) {
                    val = layer->_data[i][0][j];
                    mx_id = j;
                }
            }
            labels.push_back(mx_id);
        }

    }
};

}

#endif
