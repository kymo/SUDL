#ifndef _GRU_H_
#define _GRU_H_

#include <iostream>
#include <string.h>
#include <vector>
#include <map>
#include <math.h>
#include <time.h>
#include "matrix.h"

namespace sub_dl {

class GRU {

private:
    
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

    matrix_double _hidden_output_weights;
    matrix_double _output_bias;

    matrix_double _output_values;
    matrix_double _hidden_values;
    matrix_double _rg_values;
    matrix_double _ug_values;
    matrix_double _newh_values;
        
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

public:
    GRU();
    ~GRU() {
    }
    GRU(int feature_dim, int hidden_dim, int output_dim);

    void _set_epoch_cnt(int max_epoch_cnt) {
        _max_epoch_cnt = max_epoch_cnt;
    }

    void _set_eta(double eta) {
        _eta = eta;
    }

    void _set_clip_gra(double gra) {
        _clip_gra = gra;
    }

    double _epoch(const std::vector<int>& sample_indexes, int epoch);
    
    void _backward(const matrix_double& feature,
        const matrix_double& label);

    void _forward(const matrix_double& feature);

    void _push_feature(const matrix_double& feature,
        const matrix_double& label) {
        _train_x_features.push_back(feature);
        _train_y_labels.push_back(label);
    }

    void _load_feature_data();
    void _train();
    void _predict();
    void _save_model(const std::string& file_name);
    void _load_model(const std::string& file_name);
};

}

#endif
