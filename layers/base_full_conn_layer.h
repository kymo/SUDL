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

#ifndef BASE_FULLCONN_LAYER_H
#define BASE_FULLCONN_LAYER_H

#include "layer.h"

namespace sub_dl {

class BaseFullConnLayer : public Layer {

public:
    
	matrix_double _full_conn_weights;
    matrix_double _full_conn_bias;
    matrix_double _delta_full_conn_weights;
    matrix_double _delta_full_conn_bias;
	matrix_double _pre_layer_data;
	BaseFullConnLayer() {}
    
	BaseFullConnLayer(int input_dim, int output_dim) {
		_input_dim = input_dim;
		_output_dim = output_dim;
        _full_conn_weights.resize(input_dim, output_dim);
        _full_conn_bias.resize(1, output_dim);
        _full_conn_weights.assign_val();
        _full_conn_bias.assign_val();
		_delta_full_conn_weights.resize(input_dim, output_dim);
		_delta_full_conn_bias.resize(1, output_dim);
		_type = FULL_CONN;
    }

	void display() {
		std::cout << "--------------full conn layer-----------" << std::endl;
		_data[0]._display("data");
		if (_errors.size() > 0) 
		_errors[0]._display("error");
		std::cout << "--------------full conn layer end-----------" << std::endl;
	}

	
	void _update_gradient(int opt_type, double learning_rate) {
		if (opt_type == SGD) {
			_full_conn_weights.add(_delta_full_conn_weights * learning_rate);
			_full_conn_bias.add(_delta_full_conn_bias * learning_rate);
		}
	}

};

}
#endif
