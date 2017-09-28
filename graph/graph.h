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
#include <queue>

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
        _layer->_forward(_pre_layers[0]);
    }

    void _backward() {
        _layer->_backward(_nxt_layers[0]);
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
    std::map<int, Layer*> _layer_map;
    std::map<int, std::vector<int> > _edges;
    std::map<int, std::vector<int> > _reverse_edges;
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

    // add a node into graph, and build the edges between the node and
    // the other nodes in id_vec
    int _add_node(Layer* layer, std::vector<int> id_vec) {
        // _operators.push_back(new Operator(layer, ++_node_cnt));
        ++_node_cnt;
        
        _op_map[_node_cnt] = new Operator(layer, _node_cnt);
        
        if (_reverse_edges.find(_node_cnt) == _reverse_edges.end()) {
            _reverse_edges[_node_cnt] = std::vector<int>();
        }
        _node_ins[_node_cnt] = id_vec.size();

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

    void _forward_compute() {
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
    }

    void _backward_compute() {
        // id ==
        std::map<int, int> node_ins(_reverse_node_ins.begin(), _node_ins.end());
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
            for (auto id : _edges[fid]) {
                node_ins[id] --;
                if (0 == node_ins[id]) {
                    _queue.push(id);
                }
            }
        }
    }

};
}

#endif
