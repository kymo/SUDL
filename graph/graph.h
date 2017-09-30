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

#ifndef GRAPH_H_
#define GRAPH_H_

#include <iostream>
#include <stdio.h>
#include <string.h>
#include <stdlib.h> 
#include "util.h"
#include "matrix.h"
#include "layer.h"
#include "loss_layer.h"
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
#include <queue>
#include "sudl.pb.h"
#include "layer_factory.h"

namespace sub_dl {

typedef struct node {

    std::vector<Layer*> _pre_layers;
    std::vector<Layer*> _nxt_layers;
    
    Layer* _layer;
    int _id;
    node() {}
    node(Layer* layer, int id) : _layer(layer) , _id(id) {}

    void _add_pre_layers(Layer* _layer) {
        _pre_layers.push_back(_layer);
    }

    void _add_nxt_layer(Layer* _layer) {
        _nxt_layers.push_back(_layer);
    }

    void _forward() {
        if (_pre_layers.size() > 0) {
            _layer->_forward(_pre_layers[0]);
        }
    }

    void _backward() {
        if (_nxt_layers.size() > 0) {
            _layer->_backward(_nxt_layers[0]);
        } else {
            _layer->_backward(NULL);
        }
    }

    void _update_gradient() {
        _layer->_update_gradient(SGD, -0.001);
    }

    void _clear_gradient() {
        _layer->_clear_gradient();
    }

} Operator;

class Graph {

private:

    std::map<std::string, int> _name_id_map;

    std::map<int, std::vector<int> > _edges;
    std::map<int, std::vector<int> > _reverse_edges;
    
    std::vector<int> _input_nodes;
    std::map<int, Operator*> _op_map;
    
    std::map<int, int> _node_ins;
    std::map<int, int> _reverse_node_ins;

    std::queue<int> _queue;
    bool _visited;
    int _node_cnt;

public:    
    Graph() {
        _node_cnt = 0;
    }
    
    double _gradient_check(const std::vector<std::vector<matrix_double> >& batch_x,
        const std::vector<matrix_double>& batch_y) {

        std::vector<Layer*> _layers;
        for (auto it = _op_map.begin(); it != _op_map.end(); it++) {
            _layers.push_back(it->second->_layer);
        }

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
                for (int k = 0; k < batch_x.size(); k++) {    
                    matrix_double label;
                    label_encode(batch_y[k], label, 4);
                    _set_input(batch_x[k]);
                    _set_label(label);
                    f1 += _forward_compute();
                }
                weights[i][j] = v - 1.0e-4;
                for (int k = 0; k < batch_x.size(); k++) {
                    matrix_double label;
                    label_encode(batch_y[k], label, 4);
                    _set_input(batch_x[k]);
                    _set_label(label);
                    f2 += _forward_compute(); 
                }
                std::cout << "[ " << delta_weights[i][j] << "," << (f1 - f2) / (2.0e-4) << "]";
                weights[i][j] = v;
            }
        }
    }


    // add a node into graph, and build the edges between the node and
    // the other nodes in id_vec
    int _add_node(Layer* layer, std::vector<int> id_vec) {

        // what add is the input  layer
        if (id_vec.size() == 0) {
            ++_node_cnt;
            _input_nodes.push_back(_node_cnt);
            _op_map[_node_cnt] = new Operator(layer, _node_cnt);
            _node_ins[_node_cnt] = 0;
            return _node_cnt;
        }

        // _operators.push_back(new Operator(layer, ++_node_cnt));
        ++_node_cnt;
      
        _op_map[_node_cnt] = new Operator(layer, _node_cnt);
        
        if (_reverse_edges.find(_node_cnt) == _reverse_edges.end()) {
            _reverse_edges[_node_cnt] = std::vector<int>();
        }
        _node_ins[_node_cnt] = id_vec.size();
        _reverse_node_ins[_node_cnt] = 0;
        // build the connection from the forward direction 
        for (auto id : id_vec) {
            if (_edges.find(id) == _edges.end()) {
                _edges[id] = std::vector<int>();
            }
            _edges[id].push_back(_node_cnt);
            if (_reverse_node_ins.find(id) == _reverse_node_ins.end()) {
                _reverse_node_ins[id] = 0;
            }
            _reverse_node_ins[id] ++;
            _reverse_edges[_node_cnt].push_back(id);

            // build layer connection
            _op_map[_node_cnt]->_add_pre_layers(_op_map[id]->_layer);
            _op_map[id]->_add_nxt_layer(layer);
        }
        return _node_cnt;
    }

    void _set_input(const std::vector<matrix_double>& feature) {
        DataFeedLayer* input_layer = (DataFeedLayer*)_op_map[_input_nodes[0]]->_layer;
        input_layer->_set_data(feature);
    }

    void _set_label(const matrix_double& label) {
        LossLayer* layer = (LossLayer*) _op_map[_node_cnt]->_layer;
        layer->_set_label(label);
    }

    void _run(std::vector<std::vector<matrix_double> >& batch_x,
        std::vector<matrix_double> batch_y, int _output_dim) {

        _clear_gradient();
        for (int i = 0; i < batch_x.size(); i++) {
            matrix_double label;
            _set_input(batch_x[i]);
            label_encode(batch_y[i], label, _output_dim);
            _set_label(label);
            _forward_compute();
            _backward_compute();
        }
        _gradient_check(batch_x, batch_y);
        _update_gradient();
    }

    void _clear_gradient() {
        for (auto it = _op_map.begin(); it != _op_map.end(); it++) {
            it->second->_clear_gradient();
        }
    }

    void _update_gradient() {
        for (auto it = _op_map.begin(); it != _op_map.end(); it++) {
            it->second->_update_gradient();
        }
    }

    double _forward_compute() {
        // id=0 is the input layer
        std::map<int, int> node_ins(_node_ins.begin(), _node_ins.end());
        for (auto it = node_ins.begin(); it != node_ins.end(); it++) {
            if (it->second == 0) {
                _queue.push(it->first);
            }
        }

        while (! _queue.empty()) {
            int fid = _queue.front();
            _op_map[fid]->_forward();
            _queue.pop();
            if (_edges.find(fid) == _edges.end()) {
                continue;
            }
            for (auto id : _edges[fid]) {
                node_ins[id] --;
                if (0 == node_ins[id]) {
                    _queue.push(id);
                }
            }
        }
        
        double cost = 0.0;
        LossLayer* layer = (LossLayer*) _op_map[_node_cnt]->_layer;
        for (auto vec : layer->_data) {
            cost += vec.sum();
        }
        return cost;
    }



    void _backward_compute() {
        // id ==
        
        std::map<int, int> node_ins(_reverse_node_ins.begin(), _reverse_node_ins.end());
        
        for (auto it = node_ins.begin(); it != node_ins.end(); it++) {
            if (it->second == 0) {
                _queue.push(it->first);
            }
        }
        while (! _queue.empty()) {
            int fid = _queue.front();
            _op_map[fid]->_backward();
            _queue.pop();
            if (_reverse_edges.find(fid) == _reverse_edges.end()) {
                continue;
            }
            for (auto id : _reverse_edges[fid]) {
                node_ins[id] --;
                if (0 == node_ins[id]) {
                    _queue.push(id);
                }
            }
        }
    }

    bool _read_from_file(const char*file_name) {

        lm::Net net;
        read_proto_from_txt_file<lm::Net>(file_name, &net);
        for (int i = 0; i < net.layer_size(); i++) {
            const lm::LayerParam &layer_param = net.layer(i);
            // Layer* layer = LayerFactory::_get_instance()->_produce(layer_param);
            Layer* layer = CREATER_LAYER(layer_param);
            std::vector<int> pre_layer_ids;
            // bottoms
            for (int j = 0; j < layer_param.bottoms_size(); j++) {
                const std::string& bottom_layer_name = layer_param.bottoms(j);
                std::cout << "bottom" << bottom_layer_name << std::endl;
                if (_name_id_map.find(bottom_layer_name) == _name_id_map.end()) {
                    std::cout << "Error when read prototxt file, layer must be in order!" << std::endl;
                    return false;
                }
                pre_layer_ids.push_back(_name_id_map[bottom_layer_name]);
            }
            int id = _add_node(layer, pre_layer_ids);
            if (_name_id_map.find(layer_param.top()) != _name_id_map.end()) {
                std::cout << "Error when read prototxt file, layer top must be different!" << std::endl;
                return false;
            }
            std::cout << "TOP" << layer_param.top() << std::endl;
            _name_id_map[layer_param.top()] = id;
        }
        return true;
    }
};

}

#endif
