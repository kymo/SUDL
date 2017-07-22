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

class LSTM_OUT {

public:
    // 
    matrix_double _output_values;
    matrix_double _hidden_values;
    matrix_double _cell_values;
    matrix_double _og_values;
    matrix_double _ig_values;
    matrix_double _fg_values;
    matrix_double _cell_new_values;
    
    void _resize(int time_step_cnt,
        int hidden_dim, 
        int output_dim) {
        _output_values.resize(time_step_cnt, output_dim);
        _hidden_values.resize(time_step_cnt, hidden_dim);
        _cell_values.resize(time_step_cnt, hidden_dim);
        _ig_values.resize(time_step_cnt, hidden_dim);
        _og_values.resize(time_step_cnt, hidden_dim);
        _fg_values.resize(time_step_cnt, hidden_dim);
        _cell_new_values.resize(time_step_cnt, hidden_dim);
    }
};

class LSTM {

private:
    
    // input gate
    matrix_double _ig_input_weights;
    matrix_double _ig_hidden_weights;
    matrix_double _ig_cell_weights;
    matrix_double _ig_bias;
    
    matrix_double _ig_delta_input_weights;
    matrix_double _ig_delta_hidden_weights;
    matrix_double _ig_delta_cell_weights;
    matrix_double _ig_delta_bias;

    // forget gate
    matrix_double _fg_input_weights;
    matrix_double _fg_hidden_weights;
    matrix_double _fg_cell_weights;
    matrix_double _fg_bias;
    matrix_double _fg_delta_input_weights;
    matrix_double _fg_delta_hidden_weights;
    matrix_double _fg_delta_cell_weights;
    matrix_double _fg_delta_bias;

    // output gate
    matrix_double _og_input_weights;
    matrix_double _og_hidden_weights;
    matrix_double _og_cell_weights;
    matrix_double _og_bias;
    matrix_double _og_delta_input_weights;
    matrix_double _og_delta_hidden_weights;
    matrix_double _og_delta_cell_weights;
    matrix_double _og_delta_bias;

    // new cell state
    matrix_double _cell_input_weights;
    matrix_double _cell_hidden_weights;
    matrix_double _cell_bias;
    matrix_double _cell_delta_input_weights;
    matrix_double _cell_delta_hidden_weights;
    matrix_double _cell_delta_bias;

    // output layer
    matrix_double _hidden_output_weights;
    matrix_double _output_bias;
    matrix_double _delta_hidden_output_weights;
    matrix_double _delta_output_bias;

    std::vector<matrix_double> _x_features;
    std::vector<matrix_double> _y_labels;

    std::vector<matrix_double> _train_x_features;
    std::vector<matrix_double> _train_y_labels;
    std::vector<matrix_double> _test_x_features;
    std::vector<matrix_double> _test_y_labels;

    int _feature_dim;
    int _hidden_dim;
    int _output_dim;

    int _max_epoch_cnt;
    double _eta;
    double _clip_gra;
    bool _use_peelhole;
public:
    LSTM();
    ~LSTM() {
    }
    LSTM(int feature_dim, int hidden_dim, int output_dim, bool use_peelhole);

    void _set_epoch_cnt(int max_epoch_cnt) {
        _max_epoch_cnt = max_epoch_cnt;
    }

    void _set_eta(double eta) {
        _eta = eta;
    }

    void _set_clip_gra(double gra) {
        _clip_gra = gra;
    }

    void _set_use_peelhole(bool use_peelhole) {
        _use_peelhole = use_peelhole;
    }

    void gradient_check(matrix_double& weights, 
        matrix_double& delta_weights,
        const std::string& weights_name,
        const matrix_double& feature,
        const matrix_double& label);

    void _push_feature(const matrix_double& feature,
        const matrix_double& label) {
        _train_x_features.push_back(feature);
        _train_y_labels.push_back(label);
    }

    double _epoch(const std::vector<int>& sample_indexes, int epoch);
    
    void _backward(const matrix_double& feature,
        const matrix_double& label,
        const LSTM_OUT& lstm_layer_values);
    
    void _forward(const matrix_double& feature,
        LSTM_OUT& lstm_layer_values);

    void _load_feature_data();
    void _train();
    void _predict();
    void _save_model(const std::string& file_name);
    void _load_model(const std::string& file_name);
    
};

}

#endif
