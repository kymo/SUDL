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

#ifndef NETWORD_WRAPPER_H
#define NETWORD_WRAPPER_H

#include <iostream>
#include "seq_full_conn_layer.h"
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

template <typename T>
class NetworkWrapper {

public:
    // input dim
    int _input_dim;
    // output dim
    int _output_dim;

    NetworkWrapper(int output_dim) {
        _output_dim = output_dim;
    }

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
    void _rnn_forward(const matrix_double& feature) {
        std::vector<matrix_double> x;
        for (int i = 0; i < feature._y_dim; i++) {           
            matrix_double val(1, 1);
            val[0][0] = feature[0][i];
            x.push_back(val);
        }
        DataFeedLayer* data_layer = new DataFeedLayer(x);
        Layer* pre_layer = data_layer;
        for (auto layer : _layers) {
            layer->_forward(pre_layer);
            pre_layer = layer;
        }
        if (NULL != data_layer) {
            delete data_layer;
            data_layer = NULL;
        }
    }
    
    /* 
    * @brief 
    *    calculate output of each layer of cnn 
    *
    * @param
    *    data: the input feature maps
    *
    * @return
    */
    void _cnn_forward(const std::vector<matrix_double>& data) {
        DataFeedLayer* data_layer = new DataFeedLayer(data);
        Layer* pre_layer = data_layer;
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
            layer->_update_gradient(SGD, -0.01);
        }
    }

    void rnn_gradient_weights_check(const matrix_double& feature,
        const matrix_double& label,
        matrix_double& weights,
        const matrix_double& delta_weights) {
        for (int i = 0; i < weights._x_dim; i++) {
            for (int j = 0; j < weights._y_dim; j++) {
                
                double v = weights[i][j];
                weights[i][j] = v + 1.0e-4;
                _rnn_forward(feature);

                Layer* nxt_layer = new T(label);
                nxt_layer->_forward(_layers.back());
                nxt_layer->_backward(NULL);
                double f1 = 0.0;
                for (int k = 0; k < nxt_layer->_seq_len; k++) {
                    f1 += nxt_layer->_data[k].sum();
                }

                weights[i][j] = v - 1.0e-4;
                _rnn_forward(feature);
                nxt_layer = new T(label);
                nxt_layer->_forward(_layers.back());
                nxt_layer->_backward(NULL);
                double f2 = 0.0;
                for (int k = 0; k < nxt_layer->_seq_len; k++) {
                    f2 += nxt_layer->_data[k].sum();
                }
                std::cout << "[ " << delta_weights[i][j] << "," << (f1 - f2) / (2.0e-4) << "]";
                weights[i][j] = v;
            }
        }
    }
    
    void cnn_gradient_weights_check(const std::vector<matrix_double>& feature,
        const matrix_double& label,
        matrix_double& weights,
        const matrix_double& delta_weights) {
        for (int i = 0; i < weights._x_dim; i++) {
            for (int j = 0; j < weights._y_dim; j++) {
                
                double v = weights[i][j];
                weights[i][j] = v + 1.0e-4;
                _cnn_forward(feature);

                Layer* nxt_layer = new T(label);
                nxt_layer->_forward(_layers.back());
                nxt_layer->_backward(NULL);
                double f1 = 0.0;
                for (int k = 0; k < nxt_layer->_data.size(); k++ ) {
                    f1 += nxt_layer->_data[k].sum();
                }

                weights[i][j] = v - 1.0e-4;
                _cnn_forward(feature);
                nxt_layer = new T(label);
                nxt_layer->_forward(_layers.back());
                nxt_layer->_backward(NULL);
                double f2 = 0.0;
                for (int k = 0; k < nxt_layer->_data.size(); k++) {
                    f2 += nxt_layer->_data[k].sum();
                }
                std::cout << "[ " << delta_weights[i][j] << "," << (f1 - f2) / (2.0e-4) << "]";
                weights[i][j] = v;
            }
        }
    }

    void rnn_gradient_check(const matrix_double& feature,
        const matrix_double& label) {
        
        for (int l = 0; l < _layers.size(); l++) {
            if (_layers[l]->_type == SEQ_FULL) {
                std::cout << "------------Gradient Check for seq full layer -------------" << std::endl;
                SeqFullConnLayer* seq_full = (SeqFullConnLayer*) _layers[l];
                rnn_gradient_weights_check(feature, label, seq_full->_seq_full_weights, seq_full->_delta_seq_full_weights);
                rnn_gradient_weights_check(feature, label, seq_full->_seq_full_bias, seq_full->_delta_seq_full_bias);
            } else if (_layers[l]->_type == RNN_CELL) {
                std::cout << "------------Gradient Check for rnn cell layer -------------" << std::endl;
                RnnCell* rnn_cell = (RnnCell*) _layers[l];
                rnn_gradient_weights_check(feature, label, rnn_cell->_input_hidden_weights, rnn_cell->_delta_input_hidden_weights); 
            } else if (_layers[l]->_type == LSTM_CELL) {
                std::cout << "------------Gradient Check for lstm cell layer -------------" << std::endl;
                LstmCell* lstm_cell = (LstmCell*) _layers[l];
                rnn_gradient_weights_check(feature, label, lstm_cell->_ig_input_weights, lstm_cell->_ig_delta_input_weights);
            } else if (_layers[l]->_type == BI_LSTM_CELL) {
                std::cout << "------------Gradient Check for bi lstm cell layer -------------" << std::endl;
                LstmCell* lstm_cell = ((BiCellWrapper<LstmCell>*) _layers[l])->_pos_seq_cell;
                rnn_gradient_weights_check(feature, label, lstm_cell->_ig_input_weights, lstm_cell->_ig_delta_input_weights);
                rnn_gradient_weights_check(feature, label, lstm_cell->_og_input_weights, lstm_cell->_og_delta_input_weights);
                rnn_gradient_weights_check(feature, label, lstm_cell->_fg_input_weights, lstm_cell->_fg_delta_input_weights);
                rnn_gradient_weights_check(feature, label, lstm_cell->_cell_input_weights, lstm_cell->_cell_delta_input_weights);
            } else if (_layers[l]->_type == BI_RNN_CELL) {
                std::cout << "------------Gradient Check for bi rnn cell layer -------------" << std::endl;
                RnnCell* rnn_cell = ((BiCellWrapper<RnnCell>*) _layers[l])->_pos_seq_cell;
                rnn_gradient_weights_check(feature, label, rnn_cell->_input_hidden_weights, rnn_cell->_delta_input_hidden_weights);
                rnn_gradient_weights_check(feature, label, rnn_cell->_hidden_weights, rnn_cell->_delta_hidden_weights);
            } else if (_layers[l]->_type == BI_GRU_CELL) {
                std::cout << "------------Gradient Check for bi gru cell layer -------------" << std::endl;
                GruCell* gru_cell = ((BiCellWrapper<GruCell>*) _layers[l])->_pos_seq_cell;
                rnn_gradient_weights_check(feature, label, gru_cell->_ug_hidden_weights, gru_cell->_delta_ug_hidden_weights); 
                rnn_gradient_weights_check(feature, label, gru_cell->_rg_hidden_weights, gru_cell->_delta_rg_hidden_weights); 
                rnn_gradient_weights_check(feature, label, gru_cell->_newh_hidden_weights, gru_cell->_delta_newh_hidden_weights); 
                rnn_gradient_weights_check(feature, label, gru_cell->_newh_input_weights, gru_cell->_delta_newh_input_weights); 

            } else if (_layers[l]->_type == GRU_CELL) {
                std::cout << "------------Gradient Check for gru cell layer -------------" << std::endl;
                GruCell* gru_cell = (GruCell*) _layers[l];
                rnn_gradient_weights_check(feature, label, gru_cell->_ug_hidden_weights, gru_cell->_delta_ug_hidden_weights); 
                rnn_gradient_weights_check(feature, label, gru_cell->_rg_hidden_weights, gru_cell->_delta_rg_hidden_weights); 
                rnn_gradient_weights_check(feature, label, gru_cell->_newh_hidden_weights, gru_cell->_delta_newh_hidden_weights); 
                rnn_gradient_weights_check(feature, label, gru_cell->_newh_input_weights, gru_cell->_delta_newh_input_weights); 
            } 
        }
    }

    void cnn_gradient_check(const std::vector<matrix_double>& feature,
        const matrix_double& label) {
        
        for (int l = 0; l < _layers.size(); l++) {
            if (_layers[l]->_type == CONV) {
                ConvLayer* layer = (ConvLayer*) _layers[l];
                std::cout << "-------Conv Layer Gradient Check Result ---------" << std::endl;
                for (int i = 0; i < layer->_input_dim;i ++) {
                    for (int j = 0; j < layer->_output_dim; j++) {
                        if (layer->_conn_map[i][j]) {
                            cnn_gradient_weights_check(feature, label, 
                                layer->_conv_kernels[i][j], layer->_delta_conv_kernels[i][j]);
                         }
                    }
                }
            }
        }
    }

    double _backward(const matrix_double& label) {
        T* loss_layer = new T(label);
        loss_layer->_forward(_layers.back());
        loss_layer->_backward(NULL);
        
        Layer* nxt_layer = loss_layer;
        double cost = 0.0;
        for (int i = 0; i < loss_layer->_data.size(); i++) {
            cost += loss_layer->_data[i].sum();
        }
        for (int j = _layers.size() - 1; j >= 0; j--) {
            _layers[j]->_backward(nxt_layer);
            nxt_layer = _layers[j];
        }
        if (NULL != loss_layer) {
            delete loss_layer;
            loss_layer = NULL;
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
    double _rnn_train(const std::vector<matrix_double>& batch_x,
        const std::vector<matrix_double>& batch_y) {
        
        double cost = 0.0;
        std::string val1, val2;
        for (int i = 0; i < batch_x.size(); i++) {
            const matrix_double& feature = batch_x[i];
            const matrix_double& label_ids = batch_y[i];
            matrix_double label;
            label_encode(label_ids, label, _output_dim);
            // forward
            _rnn_forward(feature);
            cost += _backward(label);
            rnn_gradient_check(feature, label);
            _update_gradient();
        }
        return cost;
    }

    double _cnn_train(const std::vector<std::vector<matrix_double> >& batch_x,
        const std::vector<matrix_double>& batch_y) {
        double tot = 0.0;
        int v1, v2;
        for (int i = 0; i < batch_x.size(); i++) {
            _cnn_forward(batch_x[i]);
            double cost = _backward(batch_y[i]);
            v1 = merge(batch_y[i]);
            v2 = merge(_layers.back()->_data[0]);
            tot += cost;
            cnn_gradient_check(batch_x[i], batch_y[i]);
            _update_gradient();
        }
        return tot;
    }
    
    void _rnn_predict(const matrix_double& feature, std::vector<int>& labels) {
        _rnn_forward(feature);
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
            std::cout << mx_id << " ";
            labels.push_back(mx_id);
        }
        std::cout << std::endl;
    }

};

}

#endif
