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

#ifndef LAYER_H_
#define LAYER_H_

#include <iostream>
#include <map>
#include <vector>
#include "matrix.h"
#include "active_func.h"
#include "sub_log.h"
#include "sudl.pb.h"

namespace sub_dl {

enum {
    CONV = 0,        // convolutional layer
    POOL,            // pooling layer
    FULL_CONN,       // full connected layer
    INPUT,           // input layer
    LOSS,            // loss layer
    ACT,             // active layer(sigmoid,relu,tanh)
    FLAT,            // flatten layer
    EMB,             // embedding layer
    SEQ_FULL,        // sequence full connected layer
    SEQ_SOFTMAX,     // sequence softmax layer
    SEQ_ACT,         // sequence active layer
    SEQ_LOSS,        // sequence loss layer
    RNN_CELL,        // reccurent cell
    LSTM_CELL,       // lstm cell
    GRU_CELL,        // gru cell
    BI_RNN_CELL,     // bi-directional rnn cell
    BI_LSTM_CELL,    // bi-directional lstm cell
    BI_GRU_CELL,     // bi-directional gru cell
};

enum {
    SGD = 0,         // sgd
};

class Layer {

public:

    // output of the layer
    std::vector<matrix_double> _data;
    // error of the layer
    std::vector<matrix_double> _errors;
    
    // layer type
    int _type;
    
    // x_dim & y_dim of feature map
    int _feature_x_dim;
    int _feature_y_dim;
    
    // rnn: _input_dim & _outpupt_dim indicate the feature length 
    // and output length in time t
    // cnn: _input_dim & _output_dim indicate the number of input 
    // or output feature map
    int _input_dim;
    int _output_dim;

    // use only for sequence model
    int _seq_len;    

    // layer before current layer
    Layer* _pre_layer;
    // layer after current layer
    Layer* _nxt_layer;
    // active func
    ActiveFunc<double>* _active_func;

    // virtual forward function
    virtual void _forward(Layer* pre_layer) = 0;
    virtual void _backward(Layer* nxt_layer) = 0;
    virtual void _update_gradient(int opt_type, double learning_rate) = 0;
    virtual void display() = 0;
    virtual void _clear_gradient() = 0;
};

class DataFeedLayer: public Layer {

public:

    DataFeedLayer () {
        _type = INPUT;
    }
    ~DataFeedLayer() {
    }

    void _set_data(const std::vector<matrix_double>& data) {
        _data = data;
    }

    void _forward(Layer* pre_layer) {}
    void _backward(Layer* nxt_laery) {}

    void _update_gradient(int opt_type, double learning_rate) {}
    void display() {}
    void _clear_gradient() {}
};


}
#endif
