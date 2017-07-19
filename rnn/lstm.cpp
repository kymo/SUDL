/*
*@filname lstm.cpp
*/
#include "lstm.h"
#include <iostream>
#include <math.h>
#include <fstream>
#include "util.h"

namespace sub_dl {


LSTM::LSTM() {
}

LSTM::LSTM(int feature_dim, int hidden_dim, int output_dim, bool use_peelhole) : 
    _feature_dim(feature_dim), 
    _hidden_dim(hidden_dim), 
    _output_dim(output_dim) {

    _ig_input_weights.resize(feature_dim, hidden_dim);
    _ig_hidden_weights.resize(hidden_dim, hidden_dim);
    _ig_bias.resize(1, hidden_dim);

    _ig_input_weights.assign_val();
    _ig_hidden_weights.assign_val();
    _ig_bias.assign_val();

    _fg_input_weights.resize(feature_dim, hidden_dim);
    _fg_hidden_weights.resize(hidden_dim, hidden_dim);
    _fg_bias.resize(1, hidden_dim);

    _fg_input_weights.assign_val();
    _fg_hidden_weights.assign_val();
    _fg_bias.assign_val();
    
    _og_input_weights.resize(feature_dim, hidden_dim);
    _og_hidden_weights.resize(hidden_dim, hidden_dim);
    _og_bias.resize(1, hidden_dim);

    _og_input_weights.assign_val();
    _og_hidden_weights.assign_val();
    _og_bias.assign_val();

    _cell_input_weights.resize(feature_dim, hidden_dim);
    _cell_hidden_weights.resize(hidden_dim, hidden_dim);
    _cell_bias.resize(1, hidden_dim);

    _cell_input_weights.assign_val();
    _cell_hidden_weights.assign_val();
    _cell_bias.assign_val();

    _hidden_output_weights.resize(hidden_dim, output_dim);
    _output_bias.resize(1, output_dim);

    _hidden_output_weights.assign_val();
    _output_bias.assign_val();

    if (use_peelhole) {
        _fg_cell_weights.resize(hidden_dim, hidden_dim);
        _og_cell_weights.resize(hidden_dim, hidden_dim);
        _ig_cell_weights.resize(hidden_dim, hidden_dim);
    }

}

void LSTM::_forward(const matrix_double& feature,
    LSTM_OUT& lstm_layer_values) {
    
    int seq_len = feature._x_dim;
    lstm_layer_values._resize(seq_len, _hidden_dim, _output_dim);
    matrix_double pre_hidden_vals(1, _hidden_dim);
    matrix_double pre_cell_vals(1, _hidden_dim);
    
    for (int t = 0; t < seq_len; t++) {
        // xt
        const matrix_double& xt = feature._R(t);
        // fo = sigmoid(xt * _fg_input_weights + h(t-1) * _fg_hidden_weights + bias)
        matrix_double f_output;
        if (! _use_peelhole) {
            f_output = sigmoid_m(xt * _fg_input_weights 
                + pre_hidden_vals * _fg_hidden_weights + _fg_bias);
        } else {
            f_output = sigmoid_m(xt * _fg_input_weights
                + pre_hidden_vals * _fg_hidden_weights 
                + pre_cell_vals * _fg_cell_weights + _fg_bias);
        }
        // io = sigmod_m(xt * _ig_input_weights + h(t - 1) * _ig_hidden_weights + bias)
        matrix_double i_output;
        if (! _use_peelhole) {
            i_output = sigmoid_m(xt * _ig_input_weights 
                + pre_hidden_vals * _ig_hidden_weights + _ig_bias);
           } else {
            i_output = sigmoid_m(xt * _ig_input_weights 
                + pre_hidden_vals * _ig_hidden_weights 
                + pre_cell_vals * _ig_cell_weights
                + _ig_bias);
        }
        // fo = sigmoid(xt * _og_input_weights + h(t-1) * _og_hidden_weights + bias)
        matrix_double o_output;
        if (! _use_peelhole) {
            o_output = sigmoid_m(xt * _og_input_weights 
                + pre_hidden_vals * _og_hidden_weights + _og_bias);
        } else {
            o_output = sigmoid_m(xt * _og_input_weights 
                + pre_hidden_vals * _og_hidden_weights
                + pre_cell_vals * _og_cell_weights + _og_bias);
        }
        // c^~(t) = sigmoid(xt * _cell_input_weights + h(t-1) * _cell_hidden_weights) + bias
        matrix_double cell_new_val = tanh_m(xt * _cell_input_weights 
            + pre_hidden_vals * _cell_hidden_weights + _cell_bias);
        // ct = c(t-1) * fo + c^~(t) * io

        matrix_double cell_output = cell_new_val.dot_mul(i_output) + pre_cell_vals.dot_mul(f_output);
        pre_cell_vals = cell_output;
        pre_hidden_vals = tanh_m(cell_output).dot_mul(o_output);
        //
        matrix_double o_val = sigmoid_m(pre_cell_vals * _hidden_output_weights + _output_bias);
        
        lstm_layer_values._cell_values.set_row(t, pre_cell_vals);
        lstm_layer_values._output_values.set_row(t, o_val);
        lstm_layer_values._hidden_values.set_row(t, pre_hidden_vals);
        lstm_layer_values._og_values.set_row(t, o_output);
        lstm_layer_values._ig_values.set_row(t, i_output);
        lstm_layer_values._fg_values.set_row(t, f_output);
        lstm_layer_values._cell_new_values.set_row(t, cell_new_val);
    
    } 

}

void LSTM::_backward(const matrix_double& feature,
    const matrix_double& label,
    const LSTM_OUT& lstm_layer_values) {

    int seq_len = feature._x_dim;

    matrix_double nxt_hidden_error(1, _hidden_dim);
    matrix_double nxt_cell_error(1, _hidden_dim);
    
    _ig_delta_input_weights.resize(_feature_dim, _hidden_dim);
    _ig_delta_hidden_weights.resize(_hidden_dim, _hidden_dim);
    _ig_delta_bias.resize(1, _hidden_dim);
    
    _fg_delta_input_weights.resize(_feature_dim, _hidden_dim);
    _fg_delta_hidden_weights.resize(_hidden_dim, _hidden_dim);
    _fg_delta_bias.resize(1, _hidden_dim);
    
    _og_delta_input_weights.resize(_feature_dim, _hidden_dim);
    _og_delta_hidden_weights.resize(_hidden_dim, _hidden_dim);
    _og_delta_bias.resize(1, _hidden_dim);
    
    _cell_delta_input_weights.resize(_feature_dim, _hidden_dim);
    _cell_delta_hidden_weights.resize(_hidden_dim, _hidden_dim);
    _cell_delta_bias.resize(1, _hidden_dim);

    _delta_hidden_output_weights.resize(_hidden_dim, _output_dim);
    _delta_output_bias.resize(1, _output_dim);

    matrix_double output_error(1, _output_dim);
    matrix_double fg_error(1, _hidden_dim);
    matrix_double ig_error(1, _hidden_dim);
    matrix_double og_error(1, _hidden_dim);
    matrix_double new_cell_error(1, _hidden_dim);

    matrix_double nxt_fg_error(1, _hidden_dim);
    matrix_double nxt_ig_error(1, _hidden_dim);
    matrix_double nxt_og_error(1, _hidden_dim);
    matrix_double nxt_new_cell_error(1, _hidden_dim);

    matrix_double nxt_cell_mid_error(1, _hidden_dim);

    for (int t = seq_len - 1; t >= 0; t--) {

        // output layer error
        matrix_double output_error = (lstm_layer_values._output_values._R(t) - label._R(t)) \
            .dot_mul(sigmoid_m_diff(lstm_layer_values._output_values._R(t)));

        // before get the output gate/input gate/forget gate error
        // the mid error value should be calculated first
        // cell_mid_error and hidden_mid_error
        
        matrix_double hidden_mid_error = output_error * _hidden_output_weights._T() \
            + nxt_fg_error * _fg_hidden_weights._T() \
            + nxt_ig_error * _ig_hidden_weights._T() \
            + nxt_og_error * _og_hidden_weights._T() \
            + nxt_new_cell_error * _cell_hidden_weights._T();

        matrix_double cell_mid_error = hidden_mid_error
            .dot_mul(lstm_layer_values._og_values._R(t))
            .dot_mul(tanh_m_diff(lstm_layer_values._cell_values._R(t)));
        if (t != seq_len - 1) {
            cell_mid_error = cell_mid_error + nxt_cell_mid_error.dot_mul(lstm_layer_values._fg_values._R(t + 1));

        }
        if (_use_peelhole) {
            cell_mid_error = cell_mid_error + nxt_fg_error * _fg_cell_weights._T() 
                + nxt_ig_error * _ig_cell_weights._T()
                + nxt_og_error * _og_cell_weights._T();
        }
        // output gate error
        og_error = hidden_mid_error
            .dot_mul(tanh_m(lstm_layer_values._cell_values._R(t)))
            .dot_mul(sigmoid_m_diff(lstm_layer_values._og_values._R(t)));
        // input gate error
        ig_error = cell_mid_error
            .dot_mul(lstm_layer_values._cell_new_values._R(t))
            .dot_mul(sigmoid_m_diff(lstm_layer_values._ig_values._R(t)));
        // forget gate error
        fg_error.resize(0.0);
        if (t > 0) {
            fg_error = cell_mid_error
                .dot_mul(lstm_layer_values._cell_values._R(t - 1))
                .dot_mul(sigmoid_m_diff(lstm_layer_values._fg_values._R(t)));
        }
        // new cell error
        new_cell_error = cell_mid_error
            .dot_mul(lstm_layer_values._ig_values._R(t))
            .dot_mul(tanh_m_diff(lstm_layer_values._cell_new_values._R(t)));
        // add delta
        const matrix_double& xt_trans = feature._R(t)._T();
        
        _delta_hidden_output_weights.add(lstm_layer_values._hidden_values._R(t)._T() * output_error); 
        _ig_delta_input_weights.add(xt_trans * ig_error);
        _fg_delta_input_weights.add(xt_trans * fg_error);
        _og_delta_input_weights.add(xt_trans * og_error);
        _cell_delta_input_weights.add(xt_trans * new_cell_error);
        if (t > 0) {
            const matrix_double& hidden_value_pre = lstm_layer_values._hidden_values._R(t - 1)._T();
            _ig_delta_hidden_weights.add(hidden_value_pre * ig_error);
            _og_delta_hidden_weights.add(hidden_value_pre * og_error);
            _fg_delta_hidden_weights.add(hidden_value_pre * fg_error);
            _cell_delta_hidden_weights.add(hidden_value_pre * new_cell_error);
        }
        _delta_output_bias.add(output_error);
        _ig_delta_bias.add(ig_error);
        _og_delta_bias.add(og_error);
        _fg_delta_bias.add(fg_error);
        _cell_delta_bias.add(new_cell_error);

        nxt_ig_error = ig_error;
        nxt_og_error = og_error;
        nxt_fg_error = fg_error;
        nxt_new_cell_error = new_cell_error;
    }
    // weight update
    gradient_clip(_ig_delta_input_weights, _clip_gra);
    gradient_clip(_ig_delta_hidden_weights, _clip_gra);
    gradient_clip(_ig_delta_bias, _clip_gra);
    
    gradient_clip(_fg_delta_input_weights, _clip_gra);
    gradient_clip(_fg_delta_hidden_weights, _clip_gra);
    gradient_clip(_fg_delta_bias, _clip_gra);
    
    gradient_clip(_og_delta_input_weights, _clip_gra);
    gradient_clip(_og_delta_hidden_weights, _clip_gra);
    gradient_clip(_og_delta_bias, _clip_gra);
    
    gradient_clip(_cell_delta_input_weights, _clip_gra);
    gradient_clip(_cell_delta_hidden_weights, _clip_gra);
    gradient_clip(_cell_delta_bias, _clip_gra);
    
    gradient_clip(_delta_hidden_output_weights, _clip_gra);
    gradient_clip(_delta_output_bias, _clip_gra);

    
    _ig_input_weights.add(_ig_delta_input_weights * _eta);
    _ig_hidden_weights.add(_ig_delta_hidden_weights * _eta);
    _ig_bias.add(_ig_delta_bias * _eta);
    
    _fg_input_weights.add(_fg_delta_input_weights * _eta);
    _fg_hidden_weights.add(_fg_delta_hidden_weights * _eta);
    _fg_bias.add(_fg_delta_bias * _eta);
    
    _og_input_weights.add(_og_delta_input_weights * _eta);
    _og_hidden_weights.add(_og_delta_hidden_weights * _eta);
    _og_bias.add(_og_delta_bias * _eta);
    
    _cell_input_weights.add(_cell_delta_input_weights * _eta);
    _cell_hidden_weights.add(_cell_delta_hidden_weights * _eta);
    _cell_bias.add(_cell_delta_bias * _eta);

    _hidden_output_weights.add(_delta_hidden_output_weights * _eta);
    _output_bias.add(_delta_output_bias * _eta);

}

double LSTM::_epoch(const std::vector<int>& sample_indexes, int epoch) {
    double cost = 0.0;
    std::string val1, val2;
    for (size_t i = 0; i < sample_indexes.size(); i++) {
        const matrix_double& feature = _train_x_features[sample_indexes[i]];
        const matrix_double& label = _train_y_labels[sample_indexes[i]];
        _forward(feature, lstm_layer_values);
        val1 = merge(label, 1);
        val2 = merge(lstm_layer_values._output_values, 1);
        // calc error
        matrix_double diff_val = label - lstm_layer_values._output_values;
        cost += diff_val.dot_mul(diff_val).sum() * 0.5 / feature._x_dim;
        // error back propogation
        _backward(feature, label, lstm_layer_values);
    }
    if (epoch % 10 == 0) {
        std::cout << "Epoch " << epoch << " : " << cost << std::endl;
        std::cout << val1 << std::endl;
        std::cout << val2 << std::endl;
    }
    return cost / sample_indexes.size();
}

void LSTM::_load_feature_data() {
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

void LSTM::_train() {

    // load x
    _max_epoch_cnt = 100;
    int batch_size = 10;
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

