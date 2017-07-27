/*
*@filname rnn.cpp
*/
#include "rnn.h"
#include <iostream>
#include <math.h>
#include <fstream>
#include "util.h"

namespace sub_dl {

RNN::RNN() {
}

RNN::RNN(int feature_dim, int hidden_dim, int output_dim) : 
    _feature_dim(feature_dim), 
    _hidden_dim(hidden_dim), 
    _output_dim(output_dim) {
    
    _input_hidden_weights.resize(feature_dim, hidden_dim);
    _hidden_weights.resize(hidden_dim, hidden_dim);
    _hidden_output_weights.resize(hidden_dim, output_dim);
    _hidden_bias.resize(1, hidden_dim);
    _output_bias.resize(1, output_dim);
    _input_hidden_weights.assign_val();
    _hidden_weights.assign_val();
    _hidden_output_weights.assign_val();
    _hidden_bias.assign_val();
    _output_bias.assign_val();

}

void RNN::_forward(const matrix_double& feature,
    matrix_double& hidden_values, 
    matrix_double& output_values) {
        
    int seq_len = feature._x_dim;
    output_values.resize(seq_len, _output_dim);
    hidden_values.resize(seq_len, _hidden_dim);    
    matrix_double pre_hidden_vals(1, _hidden_dim);

    for (int t = 0; t < seq_len; t++) {
        // h_t
        const matrix_double& xt = feature._R(t);
        // net_h_t = xt * v + h_{t-1} * u + b_h
        
        matrix_double net_h_vals = xt * _input_hidden_weights + pre_hidden_vals * _hidden_weights + _hidden_bias;
        //net_h_vals._display("net_h_vals");
        // out_h_t = sigmoid(net_h_t)
        // clip_values(net_h_vals);
        pre_hidden_vals = tanh_m(net_h_vals);
        // output_h_t = out_h_t * W + b_o
        matrix_double o_vals = pre_hidden_vals * _hidden_output_weights + _output_bias;
        matrix_double output_vals = sigmoid_m(o_vals);
        output_values.set_row(t, output_vals);
        hidden_values.set_row(t, pre_hidden_vals);
    }

}

void RNN::_backward(const matrix_double& feature,
    const matrix_double& label,
    const matrix_double& hidden_values, 
    const matrix_double& output_values) {

    int seq_len = label._x_dim;
    // error back propogation
    matrix_double nxt_hidden_error(1, _hidden_dim);
    matrix_double delta_hidden_output_weights(_hidden_output_weights._x_dim, 
        _hidden_output_weights._y_dim);
    matrix_double delta_input_hidden_weights(_input_hidden_weights._x_dim,
        _input_hidden_weights._y_dim);
    matrix_double delta_hidden_weights(_hidden_weights._x_dim,
        _hidden_weights._y_dim);
    matrix_double delta_hidden_bias(_hidden_bias._x_dim, _hidden_bias._y_dim);
    matrix_double delta_output_bias(_output_bias._x_dim, _output_bias._y_dim);
    for (int t = seq_len - 1; t >= 0; t--) {
        // output_error = (o_t - y_t) * f'(o_t)
        matrix_double output_error = (output_values._R(t) - label._R(t)) \
            .dot_mul(sigmoid_m_diff(output_values._R(t)));
        // hidden_error = 
        // (output_error * V_jk + nxt_hidden_error * U_jr) dot_multiply f'(hidden_values[t-1])
        matrix_double hidden_error = (output_error * _hidden_output_weights._T() + 
            nxt_hidden_error * _hidden_weights._T()) \
            .dot_mul(tanh_m_diff(hidden_values._R(t)));
        //_hidden_output_weights.add(_hidden_values._R(t)._T() * output_error * eta);
        // delta_V = delta_V + hidden_values[t-1]^T * output_error
        delta_hidden_output_weights.add((hidden_values._R(t)._T() * 
            output_error));
        //_input_hidden_weights.add(feature._R(t)._T() * hidden_error * eta);
        //(feature._R(t)._T() * hidden_error)._display("feature._R(t)._T() * hidden_error");
        delta_input_hidden_weights.add(feature._R(t)._T() * hidden_error);

        if (t > 0) {
            //_hidden_weights.add(_hidden_values._R(t - 1)._T() * hidden_error * eta); 
            delta_hidden_weights.add(hidden_values._R(t - 1)._T() * hidden_error);
        }

        delta_hidden_bias.add(hidden_error);
        delta_output_bias.add(output_error);
        nxt_hidden_error = hidden_error; 
    }
    // gradient check
    std::cout << "--------------------Gradient Check ---------------" << std::endl;
    for (int i = 0; i <  _input_hidden_weights._x_dim; i++) {
        for (int j = 0; j < _input_hidden_weights._y_dim; j++) {
            double v = _input_hidden_weights[i][j];
            _input_hidden_weights[i][j] = v + 1.0e-4;
            _forward(feature, _hidden_values, _output_values);
            matrix_double diff_val = _output_values - label;
            double f1 = (diff_val.dot_mul(diff_val)).sum() * 0.5;
            _input_hidden_weights[i][j] = v - 1.0e-4;
            _forward(feature, _hidden_values, _output_values);
            diff_val = _output_values - label;
            double f2 = (diff_val.dot_mul(diff_val).sum()) * 0.5;
            std::cout << "[ " << delta_input_hidden_weights[i][j] << ", " << (f1 - f2) / (2.0e-4) << " ]";
            _input_hidden_weights[i][j] = v;
        }
        std::cout << std::endl;
    }
    std::cout << "_input_hidden_weights" << std::endl;
    for (int i = 0; i <  _hidden_weights._x_dim; i++) {
        for (int j = 0; j < _hidden_weights._y_dim; j++) {
            double v = _hidden_weights[i][j];
            _hidden_weights[i][j] = v + 1.0e-4;
            _forward(feature, _hidden_values, _output_values);
            matrix_double diff_val = _output_values - label;
            double f1 = (diff_val.dot_mul(diff_val)).sum() * 0.5;
            _hidden_weights[i][j] = v - 1.0e-4;
            _forward(feature, _hidden_values, _output_values);
            diff_val = _output_values - label;
            double f2 = (diff_val.dot_mul(diff_val).sum()) * 0.5;
            std::cout << "[ " << delta_hidden_weights[i][j] << ", " << (f1 - f2) / (2.0e-4) << " ]";
            _hidden_weights[i][j] = v;
        }
        std::cout << std::endl;
    }
    std::cout << "_input_hidden_output_weights" << std::endl;
    for (int i = 0; i <  _hidden_output_weights._x_dim; i++) {
        for (int j = 0; j < _hidden_output_weights._y_dim; j++) {
            double v = _hidden_output_weights[i][j];
            _hidden_output_weights[i][j] = v + 1.0e-4;
            _forward(feature, _hidden_values, _output_values);
            matrix_double diff_val = _output_values - label;
            double f1 = (diff_val.dot_mul(diff_val)).sum() * 0.5;
            _hidden_output_weights[i][j] = v - 1.0e-4;
            _forward(feature, _hidden_values, _output_values);
            diff_val = _output_values - label;
            double f2 = (diff_val.dot_mul(diff_val).sum()) * 0.5;
            std::cout << "[ " << delta_hidden_output_weights[i][j] << ", " << (f1 - f2) / (2.0e-4) << " ]";
            _hidden_output_weights[i][j] = v;
        }
        std::cout << std::endl;
    }
    
    /*
    gradient_clip(delta_hidden_output_weights, _clip_gra);
    gradient_clip(delta_input_hidden_weights, _clip_gra);
    gradient_clip(delta_hidden_weights, _clip_gra);
    gradient_clip(delta_hidden_bias, _clip_gra);
    gradient_clip(delta_output_bias, _clip_gra);
    */
    // weight update
    _hidden_output_weights.add(delta_hidden_output_weights * _eta);
    _input_hidden_weights.add(delta_input_hidden_weights * _eta);
    _hidden_weights.add(delta_hidden_weights * _eta);
    _hidden_bias.add(delta_hidden_bias * _eta);
    _output_bias.add(delta_output_bias * _eta);

}


double RNN::_epoch(const std::vector<int>& sample_indexes, int epoch) {

    double cost = 0.0;
    std::string val1, val2;
    //double val1, val2;
    for (size_t i = 0; i < sample_indexes.size(); i++) {
        const matrix_double& feature = _train_x_features[sample_indexes[i]];
        const matrix_double& label = _train_y_labels[sample_indexes[i]];
        _forward(feature, _hidden_values, _output_values);
        val1 = merge(label, 1);
        val2 = merge(_output_values, 1);
        //val1 = merge(label);
        //val2 = merge(_output_values);
        // calc error
        matrix_double diff_val = label - _output_values;
        cost += diff_val.dot_mul(diff_val).sum() * 0.5 / feature._x_dim;
        _backward(feature, label, _hidden_values, _output_values);
    }
    cost /= sample_indexes.size();
    if (epoch % 10 == 0) {
        std::cout << "Epoch " << epoch << ":" << cost << " " << val1 << " " << val2 << std::endl;
        std::cout << val1 << std::endl;
        std::cout << val2 << std::endl;
    }
    return cost;
}

void RNN::_load_feature_data() {
    srand( (unsigned)time( NULL ) );
    // load data
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

void RNN::_train() {

    // load x
    // _load_feature_data();
    _max_epoch_cnt = 100;
    int batch_size = 10;
    int tot = 100000;
    for (size_t epoch = 0; epoch < _max_epoch_cnt; epoch++) {
        for (int i = 0; i < tot / batch_size; i++) {
            std::vector<int> sample_indexes;
            for (int j = i * batch_size; j < (i + 1) * batch_size; j++) {
                sample_indexes.push_back(j);
            }
            _epoch(sample_indexes, i);
        }
    }
}

}


