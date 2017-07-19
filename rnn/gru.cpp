/*
*@filname gru.cpp
*/
#include "gru.h"
#include <iostream>
#include <math.h>
#include <fstream>
#include "util.h"

namespace sub_dl {


GRU::GRU() {
}

GRU::GRU(int feature_dim, int hidden_dim, int output_dim) : 
    _feature_dim(feature_dim), 
    _hidden_dim(hidden_dim), 
    _output_dim(output_dim) {
    
    _ug_input_weights.resize(_feature_dim, _hidden_dim);
    _ug_hidden_weights.resize(_hidden_dim, _hidden_dim);
    _ug_bias.resize(1, _hidden_dim);
    
    _rg_input_weights.resize(_feature_dim, _hidden_dim);
    _rg_hidden_weights.resize(_hidden_dim, _hidden_dim);
    _rg_bias.resize(1, _hidden_dim);

    _newh_input_weights.resize(_feature_dim, _hidden_dim);
    _newh_hidden_weights.resize(_hidden_dim, _hidden_dim);
    _newh_bias.resize(1, _hidden_dim);
    
    _hidden_output_weights.resize(_hidden_dim, _output_dim);
    _output_bias.resize(1, _output_dim);
    
    _ug_input_weights.assign_val();
    _ug_hidden_weights.assign_val();
    _ug_bias.assign_val();
    
    _rg_input_weights.assign_val();
    _rg_hidden_weights.assign_val();
    _rg_bias.assign_val();

    _newh_input_weights.assign_val();
    _newh_hidden_weights.assign_val();
    _newh_bias.assign_val();

    _hidden_output_weights.assign_val();
    _output_bias.assign_val();
     
}

void GRU::_forward(const matrix_double& feature) {
    int seq_len = feature._x_dim;
    matrix_double pre_hidden_vals(1, _hidden_dim);
    _ug_values.resize(seq_len, _hidden_dim);
    _rg_values.resize(seq_len, _hidden_dim);
    _newh_values.resize(seq_len, _hidden_dim);
    _output_values.resize(seq_len, _output_dim);
    _hidden_values.resize(seq_len, _hidden_dim);


    for (int t = 0; t < seq_len; t++) {
        // h_t
        const matrix_double& xt = feature._R(t);
        matrix_double ug_value = sigmoid_m(xt * _ug_input_weights
            + pre_hidden_vals * _ug_hidden_weights 
            + _ug_bias);
        matrix_double rg_value = sigmoid_m(xt * _rg_input_weights
            + pre_hidden_vals * _rg_hidden_weights
            + _rg_bias);
        matrix_double newh_value = tanh_m(xt * _newh_input_weights
            + (rg_value.dot_mul(pre_hidden_vals)) * _newh_hidden_weights
            + _newh_bias);
        matrix_double hidden_value = ug_value.dot_mul(newh_value) - (ug_value - 1).dot_mul(pre_hidden_vals) ;
        matrix_double output_value = sigmoid_m(hidden_value * _hidden_output_weights
            + _output_bias);

        pre_hidden_vals = hidden_value;
        _output_values.set_row(t, output_value);
        _hidden_values.set_row(t, hidden_value);
        _ug_values.set_row(t, ug_value);
        _rg_values.set_row(t, rg_value);
        _newh_values.set_row(t, newh_value);
    }

}

void GRU::_backward(const matrix_double& feature,
    const matrix_double& label) {
    int seq_len = label._x_dim;
    // error back propogation
    
    _delta_ug_input_weights.resize(_feature_dim, _hidden_dim);
    _delta_ug_hidden_weights.resize(_hidden_dim, _hidden_dim);
    _delta_ug_bias.resize(1, _hidden_dim);
    _delta_rg_input_weights.resize(_feature_dim, _hidden_dim);
    _delta_rg_hidden_weights.resize(_hidden_dim, _hidden_dim);
    _delta_rg_bias.resize(1, _hidden_dim);
    _delta_newh_input_weights.resize(_feature_dim, _hidden_dim);
    _delta_newh_hidden_weights.resize(_hidden_dim, _hidden_dim);
    _delta_newh_bias.resize(1, _hidden_dim);
    _delta_hidden_output_weights.resize(_hidden_dim, _output_dim);
    _delta_output_bias.resize(1, _output_dim);
    
    matrix_double nxt_hidden_error(1, _hidden_dim);
    matrix_double nxt_newh_error(1, _hidden_dim);
    matrix_double nxt_ug_error(1, _hidden_dim);
    matrix_double nxt_rg_error(1, _hidden_dim);

    for (int t = seq_len - 1; t >= 0; t--) {
        // output_error = (o_t - y_t) * f'(o_t)
        matrix_double output_error = (_output_values._R(t) - label._R(t)) \
            .dot_mul(sigmoid_m_diff(_output_values._R(t)));
        output_error._display("output_error");

        // hidden_error = 
        // (output_error * V_jk + nxt_hidden_error * U_jr) dot_multiply f'(hidden_values[t-1])
        matrix_double mid_hidden_error = output_error * _hidden_output_weights._T() 
            + nxt_ug_error * _ug_hidden_weights._T() 
            + nxt_rg_error * _rg_hidden_weights._T();
        if (t != seq_len - 1) {
            mid_hidden_error = mid_hidden_error 
                + nxt_newh_error.dot_mul(_rg_values._R(t + 1)) * _newh_hidden_weights._T()
                - nxt_hidden_error.dot_mul(_ug_values._R(t + 1) - 1);
        }
        matrix_double newh_error = mid_hidden_error
            .dot_mul(_ug_values._R(t))
            .dot_mul(tanh_m_diff(_newh_values._R(t)));

        matrix_double ug_error = mid_hidden_error.dot_mul(sigmoid_m_diff(_ug_values._R(t)));
        if (t > 0) {
            ug_error = ug_error.dot_mul(_newh_values._R(t) - _hidden_values._R(t - 1));
        } else {
            ug_error = ug_error.dot_mul(_newh_values._R(t));
        }

        matrix_double rg_error(1, _hidden_dim);
        if (t > 0) {
            rg_error = newh_error.dot_mul(_ug_values._R(t))
                .dot_mul(tanh_m_diff(_newh_values._R(t)))
                .dot_mul(_hidden_values._R(t - 1) * _newh_hidden_weights)
                .dot_mul(sigmoid_m_diff(_rg_values._R(t)));
        }
        rg_error._display("rg_error");
        _delta_ug_input_weights.add(feature._R(t)._T() * ug_error);
        _delta_ug_bias.add(ug_error);
        _delta_rg_input_weights.add(feature._R(t)._T() * rg_error);
        _delta_rg_bias.add(rg_error);
        _delta_newh_input_weights.add(feature._R(t)._T() * newh_error);
        _delta_newh_bias.add(newh_error);
        if (t > 0) {
            _delta_ug_hidden_weights.add(_hidden_values._R(t - 1)._T() * ug_error);
            _delta_rg_hidden_weights.add(_hidden_values._R(t - 1)._T() * ug_error);
            _delta_newh_hidden_weights.add((_hidden_values._R(t - 1).dot_mul(_rg_values._R(t)))._T() * ug_error);
        }
        _delta_hidden_output_weights.add(_hidden_values._R(t)._T() * output_error);
        _delta_output_bias.add(output_error);
        nxt_hidden_error = mid_hidden_error;
        nxt_ug_error = ug_error;
        nxt_rg_error = rg_error;
        nxt_newh_error = newh_error;
    }
    
    // weight update
    gradient_clip(_delta_ug_input_weights, _clip_gra);
    gradient_clip(_delta_ug_hidden_weights, _clip_gra);
    gradient_clip(_delta_ug_bias, _clip_gra);
    gradient_clip(_delta_rg_input_weights, _clip_gra);
    gradient_clip(_delta_rg_hidden_weights, _clip_gra);
    gradient_clip(_delta_rg_bias, _clip_gra);
    gradient_clip(_delta_newh_input_weights, _clip_gra);
    gradient_clip(_delta_newh_hidden_weights, _clip_gra);
    gradient_clip(_delta_newh_bias, _clip_gra);
    gradient_clip(_delta_hidden_output_weights, _clip_gra);
    gradient_clip(_delta_output_bias, _clip_gra);
    
    _ug_input_weights.add(_delta_ug_input_weights * _eta);
    _ug_hidden_weights.add(_delta_ug_hidden_weights * _eta);
    _ug_bias.add(_delta_ug_bias * _eta);
    _rg_input_weights.add(_delta_rg_input_weights * _eta);
    _rg_hidden_weights.add(_delta_rg_hidden_weights * _eta);
    _rg_bias.add(_delta_rg_bias * _eta);
    _newh_input_weights.add(_delta_newh_input_weights * _eta);
    _newh_hidden_weights.add(_delta_newh_hidden_weights * _eta);
    _newh_bias.add(_delta_newh_bias * _eta);
    _hidden_output_weights.add(_delta_hidden_output_weights * _eta);
    _output_bias.add(_delta_output_bias * _eta);

}


double GRU::_epoch(const std::vector<int>& sample_indexes, int epoch) {

    double cost = 0.0;
    std::string val1, val2;
    // double val1, val2;
    for (size_t i = 0; i < sample_indexes.size(); i++) {
        const matrix_double& feature = _train_x_features[sample_indexes[i]];
        const matrix_double& label = _train_y_labels[sample_indexes[i]];
        _forward(feature);
        val1 = merge(label, 1);
        val2 = merge(_output_values, 1);
        // calc error
        // val1 = merge(label);
        // val2 = merge(_output_values);
        matrix_double diff_val = label - _output_values;
        cost += diff_val.dot_mul(diff_val).sum() * 0.5 / feature._x_dim;
        _backward(feature, label);
    }
    cost /= sample_indexes.size();
    if (epoch % 10 == 0) {
        std::cout << "Epoch " << epoch << ":" << cost << " " << val1 << " " << val2 << std::endl;
        std::cout << val1 << std::endl;
        std::cout << val2 << std::endl;
    }
    return cost;
}

void GRU::_load_feature_data() {
    srand( (unsigned)time( NULL ) );
    // load data
    std::ifstream fis("data");
    for (size_t i = 0; i < 12500; i++) {    
        matrix_double x(8, _feature_dim);
        matrix_double y(8, _output_dim);
        int sum = 0;
        for (size_t j = 0; j < _feature_dim; j ++) {
            int v = rand() % 128;
            //int v;
            //fis >> v;
            sum += v;
            size_t k = 0;
            while (v > 0) {
                x[k][j] = v % 2;
                v /= 2;
                k += 1;
            }
            for (; k < 8;k ++) {
                x[k][j] = 0;
            }
        }
        //fis >> sum;
        size_t k = 0;
        while (sum > 0) {
            y[k][0] = sum % 2;
            sum /= 2;
            k += 1;
        }
        for (; k < 8;k ++) {
            y[k][0] = 0;
        }
        if (i < 10000) {
            _train_x_features.push_back(x);
            _train_y_labels.push_back(y);
        } else {
            _test_x_features.push_back(x);
            _test_y_labels.push_back(y);
        }
    }

}

void GRU::_train() {

    // load x
    // _load_feature_data();
    _max_epoch_cnt = 100;
    int batch_size = 1;
    for (size_t epoch = 0; epoch < _max_epoch_cnt; epoch++) {
        for (int i = 0; i < 10000; i++) {
            std::vector<int> sample_indexes;
            for (int j = i * batch_size; j < (i + 1) * batch_size; j++) {
                sample_indexes.push_back(j);
            }
            _epoch(sample_indexes, i);
        }
    }
}

}
