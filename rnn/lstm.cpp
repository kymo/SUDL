/*
*@filname lstm.cpp
*/
#include "lstm.h"
#include <iostream>
#include <math.h>
#include <fstream>

namespace sub_dl {

template <typename T>
void gradident_clip(Matrix<T>& matrix) {
    for (size_t i = 0; i < matrix._x_dim; i++) {
        for (size_t j = 0; j < matrix._y_dim; j++) {
            if (matrix[i][j] > 5.0 || matrix[i][j] < -5.0) {
                matrix[i][j] /= 10.0;
            }
        }
    }
}

float tanh(float x) {
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
}

float sigmoid(float x) {
    return 1.0 / (1 + exp(-x));
}

template <typename T>
Matrix<T> tanh_m(const Matrix<T>& matrix) {
    Matrix<T> ret_val(matrix._x_dim, matrix._y_dim);
    for (size_t i = 0; i < matrix._x_dim; i++) {
        for (size_t j = 0; j < matrix._y_dim; j++) {
            ret_val[i][j] = tanh(matrix[i][j]);
        }
    }
    return ret_val;
}

template <typename T>
Matrix<T> tanh_m_diff(const Matrix<T>& matrix) {
    Matrix<T> ret_val(matrix._x_dim, matrix._y_dim);
    for (size_t i = 0; i < matrix._x_dim; i++) {
        for (size_t j = 0; j < matrix._y_dim; j++) {
            ret_val[i][j] = 1 - tanh(matrix[i][j]) * tanh(matrix[i][j]);
        }
    }
    return ret_val;
}

template <typename T>
Matrix<T> sigmoid_m(const Matrix<T>& matrix) {
    Matrix<T> ret_val(matrix._x_dim, matrix._y_dim);
    for (size_t i = 0; i < matrix._x_dim; i++) {
        for (size_t j = 0; j < matrix._y_dim; j++) {
            ret_val[i][j] = sigmoid(matrix[i][j]);
        }
    }
    return ret_val;
}

template <typename T>
Matrix<T> sigmoid_m_diff(const Matrix<T>& matrix) {
    Matrix<T> ret_val(matrix._x_dim, matrix._y_dim);
    for (size_t i = 0; i < matrix._x_dim; i++) {
        for (size_t j = 0; j < matrix._y_dim; j++) {
            ret_val[i][j] = matrix[i][j] * (1 - matrix[i][j]);
        }
    }
    return ret_val;
}

int merge(const Matrix<float>& output_val) {
    int val = 0;
    int d = 1;
    for (int i = 8; i >= 1; i--) {
        val += int(output_val[i - 1][0] + 0.5) * pow(2, i - 1);
    }
    return val;
}


LSTM::LSTM() {
}

LSTM::LSTM(int feature_dim, int hidden_dim, int output_dim) : 
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

}

void LSTM::_forward(const matrix_float& feature,
    LSTM_OUT& lstm_layer_values) {
    
    int time_step_cnt = feature._x_dim;
    lstm_layer_values._resize(time_step_cnt, _hidden_dim, _output_dim);
    matrix_float pre_hidden_vals(1, _hidden_dim);
    matrix_float pre_cell_vals(1, _hidden_dim);
    
    for (int t = 0; t < time_step_cnt; t++) {
        // xt
        const matrix_float& xt = feature._R(t);
        // fo = sigmoid(xt * _fg_input_weights + h(t-1) * _fg_hidden_weights + bias)
        matrix_float f_output = sigmoid_m(xt * _fg_input_weights 
            + pre_hidden_vals * _fg_hidden_weights + _fg_bias);
        // io = sigmod_m(xt * _ig_input_weights + h(t - 1) * _ig_hidden_weights + bias)
        matrix_float i_output = sigmoid_m(xt * _ig_input_weights 
            + pre_hidden_vals * _ig_hidden_weights + _ig_bias);
        // fo = sigmoid(xt * _og_input_weights + h(t-1) * _og_hidden_weights + bias)
        matrix_float o_output = sigmoid_m(xt * _og_input_weights 
            + pre_hidden_vals * _og_hidden_weights + _og_bias);
        // c^~(t) = sigmoid(xt * _cell_input_weights + h(t-1) * _cell_hidden_weights) + bias
        matrix_float cell_new_val = tanh_m(xt * _cell_input_weights 
            + pre_hidden_vals * _cell_hidden_weights + _cell_bias);
        // ct = c(t-1) * fo + c^~(t) * io

        matrix_float cell_output = cell_new_val.dot_mul(i_output) + pre_cell_vals.dot_mul(f_output);
        pre_cell_vals = cell_output;
        pre_hidden_vals = tanh_m(cell_output).dot_mul(o_output);
        //
        matrix_float o_val = sigmoid_m(pre_cell_vals * _hidden_output_weights + _output_bias);


        lstm_layer_values._cell_values.set_row(t, pre_cell_vals);
        lstm_layer_values._output_values.set_row(t, o_val);
        lstm_layer_values._hidden_values.set_row(t, pre_hidden_vals);
        lstm_layer_values._og_values.set_row(t, o_output);
        lstm_layer_values._ig_values.set_row(t, i_output);
        lstm_layer_values._fg_values.set_row(t, f_output);
        lstm_layer_values._cell_new_values.set_row(t, cell_new_val);
    
    } 

}

void LSTM::_backward(const matrix_float& feature,
    const matrix_float& label,
    const LSTM_OUT& lstm_layer_values) {

    int time_step_cnt = feature._x_dim;

    matrix_float nxt_hidden_error(1, _hidden_dim);
    matrix_float nxt_cell_error(1, _hidden_dim);
    
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

    matrix_float output_error(1, _output_dim);
    matrix_float fg_error(1, _hidden_dim);
    matrix_float ig_error(1, _hidden_dim);
    matrix_float og_error(1, _hidden_dim);
    matrix_float new_cell_error(1, _hidden_dim);

    matrix_float nxt_fg_error(1, _hidden_dim);
    matrix_float nxt_ig_error(1, _hidden_dim);
    matrix_float nxt_og_error(1, _hidden_dim);
    matrix_float nxt_new_cell_error(1, _hidden_dim);

    matrix_float nxt_cell_mid_error(1, _hidden_dim);

    for (int t = time_step_cnt - 1; t >= 0; t--) {

        // output layer error
        matrix_float output_error = (lstm_layer_values._output_values._R(t) - label._R(t)) \
            .dot_mul(sigmoid_m_diff(lstm_layer_values._output_values._R(t)));

        // before get the output gate/input gate/forget gate error
        // the mid error value should be calculated first
        // cell_mid_error and hidden_mid_error
        
        matrix_float hidden_mid_error = output_error * _hidden_output_weights._T() \
            + nxt_fg_error * _fg_hidden_weights._T() \
            + nxt_ig_error * _ig_hidden_weights._T() \
            + nxt_og_error * _og_hidden_weights._T() \
            + nxt_new_cell_error * _cell_hidden_weights._T();

        matrix_float cell_mid_error = hidden_mid_error
            .dot_mul(lstm_layer_values._og_values._R(t))
            .dot_mul(tanh_m_diff(lstm_layer_values._cell_values._R(t))) 
            + nxt_cell_mid_error.dot_mul(lstm_layer_values._fg_values._R(t));

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
        const matrix_float& xt_trans = feature._R(t)._T();
        
        _delta_hidden_output_weights.add(lstm_layer_values._hidden_values._R(t)._T() * output_error); 
        _ig_delta_input_weights.add(xt_trans * ig_error);
        _fg_delta_input_weights.add(xt_trans * fg_error);
        _og_delta_input_weights.add(xt_trans * og_error);
        _cell_delta_input_weights.add(xt_trans * new_cell_error);
        if (t > 0) {
            const matrix_float& hidden_value_pre = lstm_layer_values._hidden_values._R(t - 1)._T();
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

float LSTM::_epoch(const std::vector<int>& sample_indexes, int epoch) {

    float cost = 0.0;
    int val1, val2;
    for (size_t i = 0; i < sample_indexes.size(); i++) {
        const matrix_float& feature = _train_x_features[sample_indexes[i]];
        const matrix_float& label = _train_y_labels[sample_indexes[i]];

        _forward(feature, lstm_layer_values);

        val1 = merge(label);
        val2 = merge(lstm_layer_values._output_values);
        float eta = -0.1;
        // calc error
        matrix_float diff_val = label - lstm_layer_values._output_values;
        cost += diff_val.dot_mul(diff_val).sum() * 0.5;
        // error back propogation
           _backward(feature, label, lstm_layer_values);
        /*
        gradident_clip(_hidden_output_weights);
        gradident_clip(_input_hidden_weights);
        gradident_clip(_hidden_weights);
        gradident_clip(_hidden_bias);
        gradident_clip(_output_bias);
        */
    }
    std::cout << "Epoch " << val1 << " " << val2 << " " << cost << std::endl;
    return cost / sample_indexes.size();
}

void LSTM::_load_feature_data() {
    srand( (unsigned)time( NULL ) );
    // load data
    std::ifstream fis("data");
    for (size_t i = 0; i < 12500; i++) {    
        matrix_float x(8, _feature_dim);
        matrix_float y(8, _output_dim);
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
    _load_feature_data();
    _max_epoch_cnt = 10;
    int batch_size = 100;
    for (size_t epoch = 0; epoch < _max_epoch_cnt; epoch++) {
        for (int i = 0; i < 100; i++) {
            std::vector<int> sample_indexes;
            for (int j = i * batch_size; j < (i + 1) * batch_size; j++) {
                sample_indexes.push_back(j);
            }
            _epoch(sample_indexes, i);
        }
    }
}

}

using namespace sub_dl;

int main() {
    LSTM *lstm = new LSTM(2, 16, 1);
    lstm->_set_epoch_cnt(100);
    lstm->_set_eta(-0.1);
    lstm->_load_feature_data();
    lstm->_train();
}
