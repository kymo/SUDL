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

#ifndef GRU_CELL_H
#define GRU_CELL_H

#include "layer.h"

namespace sub_dl {

class LstmCell;
class RnnCell;

class GruCell : public Layer {

public:
    
    matrix_double _ug_input_weights;
    matrix_double _ug_hidden_weights;
    matrix_double _ug_bias;
    
    matrix_double _delta_ug_input_weights;
    matrix_double _delta_ug_hidden_weights;
    matrix_double _delta_ug_bias;
    
    matrix_double _rg_input_weights;
    matrix_double _rg_hidden_weights;
    matrix_double _rg_bias;
   
    matrix_double _delta_rg_input_weights;
    matrix_double _delta_rg_hidden_weights;
    matrix_double _delta_rg_bias;

    matrix_double _newh_input_weights;
    matrix_double _newh_hidden_weights;
    matrix_double _newh_bias;
    
    matrix_double _delta_newh_input_weights;
    matrix_double _delta_newh_hidden_weights;
    matrix_double _delta_newh_bias;

    matrix_double _hidden_values;
    matrix_double _rg_values;
    matrix_double _ug_values;
    matrix_double _newh_values;
    
    double _eta;
    double _clip_gra;
    
    std::vector<matrix_double> _rg_errors;
    std::vector<matrix_double> _ug_errors;
    std::vector<matrix_double> _newh_errors;

    std::vector<matrix_double> _pre_layer_data;

    GruCell(int input_dim, int output_dim);

    /* 
    * @brief forward function for gru cell
    *
    * @param
    *    pre_layer: layer before gru_cell
    *        the type of legal layers are:
    *            RNN_CELL INPUT GRU_CELL GRU_CELL
    *
    * @return
    *    void
    *
    */
    void _forward(Layer* pre_layer);

    void _backward(Layer* nxt_layer);

    void display() {}

    void _update_gradient(int opt_type, double learning_rate);

    void _clear_gradient() {
        _delta_ug_input_weights.resize(0.0);
        _delta_ug_hidden_weights.resize(0.0);
        _delta_ug_bias.resize(0.0);
        _delta_rg_input_weights.resize(0.0);
        _delta_rg_hidden_weights.resize(0.0);
        _delta_rg_bias.resize(0.0);
        _delta_newh_input_weights.resize(0.0);
        _delta_newh_hidden_weights.resize(0.0);
        _delta_newh_bias.resize(0.0);
    }
};

}

#endif
