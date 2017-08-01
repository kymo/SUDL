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

#include "full_conn_layer.h"

namespace sub_dl {

void FullConnLayer::_forward(Layer* _pre_layer) {
    if (_pre_layer->_data.size() == 0 ||
        _pre_layer->_data[0]._x_dim > 1) {
        std::cerr << "Wrong Layer type before fullconnect layer!" << std::endl;
        exit(1);
    }

    std::vector<matrix_double>().swap(_data);
    _pre_layer_data = _pre_layer->_data[0];
    _data.push_back(_pre_layer_data
        * _full_conn_weights
        + _full_conn_bias);
}

void FullConnLayer::_backward(Layer* nxt_layer) {    
    if (nxt_layer->_type == CONV || nxt_layer->_type == POOL) {
        std::cerr << "Conv or Pooling before full-connected not supported yet!" << std::endl;
        exit(1);
    }
    std::vector<matrix_double>().swap(_errors);
    matrix_double error;
    _errors = nxt_layer->_errors;
    _delta_full_conn_weights.add(_pre_layer_data._T() * _errors[0]);
    _delta_full_conn_bias.add(_errors[0]);
}
}
