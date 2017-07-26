#ifndef LSTM_CELL_H
#define LSTM_CELL_H

#include "layer.h"

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

class LstmCell : public Layer {

public:
    
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

    // errors
    std::vector<matrix_double> _fg_errors;
    std::vector<matrix_double> _ig_errors;
    std::vector<matrix_double> _og_errors;
    std::vector<matrix_double> _new_cell_errors;
    
    bool _use_peelhole;
    double _eta;
    double _clip_gra;

    LstmCell(int input_dim, int output_dim, bool use_peelhole) {
        _input_dim = input_dim;
        _output_dim = output_dim;

        _type = LSTM_CELL;

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

        _use_peelhole = use_peelhole;
        
        if (use_peelhole) {
            _fg_cell_weights.resize(hidden_dim, hidden_dim);
            _og_cell_weights.resize(hidden_dim, hidden_dim);
            _ig_cell_weights.resize(hidden_dim, hidden_dim);
        }

    }

    /* 
    * @brief forward function fo basic rnn cell with tanh
    *
    * @param
    *    pre_layer: layer before rnn_cell
    *        the type of legal layers are:
    *            RNN_CELL INPUT LSTM_CELL GRU_CELL
    *
    * @return
    *    void
    *
    */
    void _forward(Layer* pre_layer) {
        std::vector<matrix_double>().swap(_data);
        _seq_len = pre_layer->_data.size();
        matrix_double pre_hidden_vals(1, _output_dim);
        matrix_double pre_cell_vals(1, _output_dim);
        for (int t = 0; t < _seq_len; t++) { 
            const matrix_double& xt = pre_layer->_data[t];
            
            matrix_double f_output;
            if (! _use_peelhole) {
                f_output = sigmoid_m(xt * _fg_input_weights 
                    + pre_hidden_vals * _fg_hidden_weights + _fg_bias);
            } else {
                f_output = sigmoid_m(xt * _fg_input_weights
                    + pre_hidden_vals * _fg_hidden_weights 
                    + pre_cell_vals * _fg_cell_weights + _fg_bias);
            }
        
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
        
            matrix_double o_output;
            if (! _use_peelhole) {
                o_output = sigmoid_m(xt * _og_input_weights 
                    + pre_hidden_vals * _og_hidden_weights + _og_bias);
            } else {
                o_output = sigmoid_m(xt * _og_input_weights 
                    + pre_hidden_vals * _og_hidden_weights
                    + pre_cell_vals * _og_cell_weights + _og_bias);
            }
        
            matrix_double cell_new_val = tanh_m(xt * _cell_input_weights 
                + pre_hidden_vals * _cell_hidden_weights + _cell_bias);
        
            matrix_double cell_output = cell_new_val.dot_mul(i_output) + pre_cell_vals.dot_mul(f_output);
            pre_cell_vals = cell_output;
            pre_hidden_vals = tanh_m(cell_output).dot_mul(o_output);
            //
            _lstm_layer_values._cell_values.set_row(t, cell_output);
            _lstm_layer_values._hidden_values.set_row(t, pre_hidden_vals);
            _lstm_layer_values._og_values.set_row(t, o_output);
            _lstm_layer_values._ig_values.set_row(t, i_output);
            _lstm_layer_values._fg_values.set_row(t, f_output);
            _lstm_layer_values._cell_new_values.set_row(t, cell_new_val);
            
        }
        _pre_layer = pre_layer;
    }

    void _backward(Layer* nxt_layer) {
        if (nxt_layer->_type != SEQ_FULL && nxt_layer->_type != RNN_CELL) {
            exit(1);
        }
        std::vector<matrix_double>().swap(_errors);
        
        int seq_len = feature._x_dim;

        _ig_delta_input_weights.resize(_feature_dim, _hidden_dim);
        _ig_delta_hidden_weights.resize(_hidden_dim, _hidden_dim);
        _ig_delta_cell_weights.resize(_hidden_dim, _hidden_dim);
        _ig_delta_bias.resize(1, _hidden_dim);
        
        _fg_delta_input_weights.resize(_feature_dim, _hidden_dim);
        _fg_delta_hidden_weights.resize(_hidden_dim, _hidden_dim);
        _fg_delta_cell_weights.resize(_hidden_dim, _hidden_dim);
        _fg_delta_bias.resize(1, _hidden_dim);
        
        _og_delta_input_weights.resize(_feature_dim, _hidden_dim);
        _og_delta_hidden_weights.resize(_hidden_dim, _hidden_dim);
        _og_delta_cell_weights.resize(_hidden_dim, _hidden_dim);
        _og_delta_bias.resize(1, _hidden_dim);
        
        _cell_delta_input_weights.resize(_feature_dim, _hidden_dim);
        _cell_delta_hidden_weights.resize(_hidden_dim, _hidden_dim);
        _cell_delta_bias.resize(1, _hidden_dim);

        _delta_hidden_output_weights.resize(_hidden_dim, _output_dim);
        _delta_output_bias.resize(1, _output_dim);

        matrix_double fg_error(1, _hidden_dim);
        matrix_double ig_error(1, _hidden_dim);
        matrix_double og_error(1, _hidden_dim);
        matrix_double new_cell_error(1, _hidden_dim);

        matrix_double nxt_fg_error(1, _hidden_dim);
        matrix_double nxt_ig_error(1, _hidden_dim);
        matrix_double nxt_og_error(1, _hidden_dim);
        matrix_double nxt_new_cell_error(1, _hidden_dim);
        matrix_double nxt_cell_mid_error(1, _hidden_dim);
        matrix_double nxt_hidden_error(1, _output_dim);
        
        matrix_double pre_layer_weights;
        std::vector<matrix_double> nxt_layer_error_weights;
        if (nxt_layer->_type == SEQ_FULL) {
            SeqFullConnLayer* seq_full_layer = (SeqFullConnLayer*) nxt_layer;
            for (int i = 0; i < _seq_len; i++) {
                nxt_layer_error_weights.push_back(nxt_layer->_errors[i]
                    * seq_full_layer->_seq_full_weights._T());
            }
        } else if (nxt_layer->_type == RNN_CELL) {
            RnnCell* rnn_cell = (RnnCell*) nxt_layer;
            for (int i = 0; i < _seq_len; i++) {
                nxt_layer_error_weights.push_back(rnn_cell->_errors[i] 
                    * rnn_cell->_input_hidden_weights._T());
            }
        } else if (nxt_layer->_type == LSTM_CELL) {
            LstmCell* lstm_cell = (LstmCell*) nxt_layer;
            for (int i = 0; i < _seq_len; i++) {
                nxt_layer_error_weights.push_back(lstm_cell->_fg_errors[i] * lstm_cell->_fg_input_weights._T()
                    + lstm_cell->_ig_errors[i] * lstm_cell->_ig_input_weights._T()
                    + lstm_cell->_og_errors[i] * lstm_cell->_og_input_weights._T()
                    + lstm_cell->_new_cell_errors[i] * lstm_cell->_cell_input_weights._T());
            }
        }

        for (int t = _seq_len - 1; t >= 0; t--) {
            
            // before get the output gate/input gate/forget gate error
            // the mid error value should be calculated first
            // cell_mid_error and hidden_mid_error

            matrix_double hidden_mid_error = nxt_layer_error_weights[t] 
                + nxt_fg_error * _fg_hidden_weights._T() \
                + nxt_ig_error * _ig_hidden_weights._T() \
                + nxt_og_error * _og_hidden_weights._T() \
                + nxt_new_cell_error * _cell_hidden_weights._T();

            matrix_double cell_mid_error = hidden_mid_error
                .dot_mul(lstm_layer_values._og_values._R(t))
                .dot_mul(tanh_m_diff(tanh_m(lstm_layer_values._cell_values._R(t))));
            
            if (t + 1 < seq_len) {
                cell_mid_error = cell_mid_error 
                    + nxt_cell_mid_error.dot_mul(lstm_layer_values._fg_values._R(t + 1));
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

            _fg_errors.push_back(fg_error);
            _ig_errors.push_back(ig_error);
            _og_errors.push_back(og_error);
            _new_cell_errors.push_back(new_cell_error);

            const matrix_double& xt_trans = feature._R(t)._T();
            
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
                
                if (_use_peelhole) {
                    const matrix_double& cell_value_pre = lstm_layer_values._cell_values._R(t - 1)._T();
                    _ig_delta_cell_weights.add(cell_value_pre * ig_error);
                    _og_delta_cell_weights.add(cell_value_pre * og_error);
                    _fg_delta_cell_weights.add(cell_value_pre * fg_error);
                }
            }
            
            _ig_delta_bias.add(ig_error);
            _og_delta_bias.add(og_error);
            _fg_delta_bias.add(fg_error);
            _cell_delta_bias.add(new_cell_error);
            nxt_ig_error = ig_error;
            nxt_og_error = og_error;
            nxt_fg_error = fg_error;
            nxt_new_cell_error = new_cell_error;
            nxt_cell_mid_error = cell_mid_error;

        }
        std::reverse(_fg_errors.begin(), _fg_errors.end());
        std::reverse(_ig_errors.begin(), _ig_errors.end());
        std::reverse(_og_errors.begin(), _og_errors.end());
        std::reverse(_new_cell_errors.begin(), _new_cell_errors.end());
    }

    void display() {
    }

    void _update_gradient(int opt_type, double learning_rate) {
        if (opt_type == SGD) {
            _ig_input_weights.add(_ig_delta_input_weights * learning_rate);
            _ig_hidden_weights.add(_ig_delta_hidden_weights * learning_rate);
            _ig_bias.add(_ig_delta_bias * learning_rate);
            
            _fg_input_weights.add(_fg_delta_input_weights * learning_rate);
            _fg_hidden_weights.add(_fg_delta_hidden_weights * learning_rate);
            _fg_bias.add(_fg_delta_bias * learning_rate);
            
            _og_input_weights.add(_og_delta_input_weights * learning_rate);
            _og_hidden_weights.add(_og_delta_hidden_weights * learning_rate);
            _og_bias.add(_og_delta_bias * learning_rate);
            
            _cell_input_weights.add(_cell_delta_input_weights * learning_rate);
            _cell_hidden_weights.add(_cell_delta_hidden_weights * learning_rate);
            _cell_bias.add(_cell_delta_bias * learning_rate);

            if (_use_peelhole) {
                _ig_cell_weights.add(_ig_delta_cell_weights * learning_rate);
                _fg_cell_weights.add(_fg_delta_cell_weights * learning_rate);
                _og_cell_weights.add(_og_delta_cell_weights * learning_rate);
            }
            
        }
    }

};

}

#endif
