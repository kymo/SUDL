#ifndef _RNN_H_
#define _RNN_H_

#include <iostream>
#include <string.h>
#include <vector>
#include <map>
#include <math.h>
#include <time.h>
#include "matrix.h"

namespace sub_dl {

class RNN {

private:
    matrix_float _input_hidden_weights;
    matrix_float _hidden_weights;
    matrix_float _hidden_output_weights;
    matrix_float _hidden_bias;
    matrix_float _output_bias;

    matrix_float _hidden_errors;
    matrix_float _hidden_values;

    matrix_float _output_errors;
    matrix_float _output_values;

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
    RNN();
    ~RNN() {
    }
    RNN(int feature_dim, int hidden_dim, int output_dim);

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
