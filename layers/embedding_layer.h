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

};

}
