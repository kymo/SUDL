#ifndef _LSTM_H_
#define _LSTM_H_

#include <iostream>
#include <string.h>
#include <vector>
#include <map>
#include <math.h>
#include <time.h>
#include "matrix.h"

namespace sub_dl {

class LSTM {

private:
    
    // input gate
    matrix_float _ig_input_weights;
    matrix_float _ig_hidden_weights;
    matrix_float _ig_cell_weights;
    matrix_float _ig_bias;
    matrix_float _ig_delta_input_weights;
    matrix_float _ig_delta_hidden_weights;
    matrix_float _ig_delta_cell_weights;
    matrix_float _ig_delta_bias;


    // forget gate
    matrix_float _fg_input_weights;
    matrix_float _fg_hidden_weights;
    matrix_float _fg_cell_weights;
    matrix_float _fg_bias;
    matrix_float _fg_delta_input_weights;
    matrix_float _fg_delta_hidden_weights;
    matrix_float _fg_delta_cell_weights;
    matrix_float _fg_delta_bias;

    // output gate
    matrix_float _og_input_weights;
    matrix_float _og_hidden_weights;
    matrix_float _og_cell_weights;
    matrix_float _og_bias;
    matrix_float _og_delta_input_weights;
    matrix_float _og_delta_hidden_weights;
    matrix_float _og_delta_cell_weights;
    matrix_float _og_delta_bias;

    // new cell state
    matrix_float _cell_input_weights;
    matrix_float _cell_hidden_weights;
    matrix_float _cell_bias;
    matrix_float _cell_delta_input_weights;
    matrix_float _cell_delta_hidden_weights;
    matrix_float _cell_delta_bias;

    // output layer
    matrix_float _hidden_output_weights;
    matrix_float _output_bias;
    matrix_float _delta_hidden_output_weights;
    matrix_float _delta_output_bias;

    // 
    matrix_float _output_values;
    matrix_float _hidden_values;
    matrix_float _cell_values;
    matrix_float _og_values;
    matrix_float _ig_values;
    matrix_float _fg_values;
    matrix_float _cell_new_values;

    

    std::vector<matrix_float> _x_features;
    std::vector<matrix_float> _y_labels;

    std::vector<matrix_float> _train_x_features;
    std::vector<matrix_float> _train_y_labels;
    std::vector<matrix_float> _test_x_features;
    std::vector<matrix_float> _test_y_labels;

    int _feature_dim;
    int _hidden_dim;
    int _output_dim;

    int _max_epoch_cnt;

public:
    LSTM();
    ~LSTM() {
    }
    LSTM(int feature_dim, int hidden_dim, int output_dim);

    void _set_epoch_cnt(int max_epoch_cnt) {
        _max_epoch_cnt = max_epoch_cnt;
    }

    float _forward(const std::vector<int>& sample_indexes, int epoch);
    void _backward();
    void _load_feature_data();
    void _train();
    void _predict();
    void _save_model(const std::string& file_name);
    void _load_model(const std::string& file_name);
};

}

#endif
