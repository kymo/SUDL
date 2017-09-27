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

#include "full_conn_softmax_layer.h"

namespace sub_dl {

void FullConnSoftmaxLayer::_forward(Layer* _pre_layer) {
    std::vector<matrix_double>().swap(_data);
    _pre_layer_data = _pre_layer->_data[0];
    std::vector<matrix_double>().swap(_data);
    matrix_double val = exp_m(_pre_layer_data * _full_conn_weights);
    double tot_val = val.sum();
    _data.push_back(val / tot_val);
}

// in this full conn softmax layer, the error from the loss layer
// has been multiplied the aj.
void FullConnSoftmaxLayer::_backward(Layer* nxt_layer) {

    std::vector<matrix_double>().swap(_errors);
    if (nxt_layer->_type == CONV || nxt_layer->_type == POOL) {
        std::cerr << "Conv or Pooling before full-connected not supported yet!" << std::endl;
        exit(1);
    }
	// double val = (_data[0].dot_mul(nxt_layer->_errors[0])).sum();
    matrix_double error = nxt_layer->_errors[0] - _data[0] * nxt_layer->_errors[0].sum();
    error = error.dot_mul(_data[0]);
	// matrix_double error = _data[0].dot_mul(nxt_layer->_errors[0] - val); 
	_errors.push_back(error);
    _delta_full_conn_weights.add(_pre_layer_data._T() * _errors[0]);
    _delta_full_conn_bias.add(_errors[0]);

}

}
