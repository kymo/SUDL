#ifndef SEQ_FULL_CONN_LAYER_
#define SEQ_FULL_CONN_LAYER_

#include "layer.h"

namespace sub_dl {

class SeqFullConnLayer : public Layer {

public:

    matrix_double _seq_full_weights;
    matrix_double _seq_full_bias;
    matrix_double _delta_seq_full_weights;
    matrix_double _delta_seq_full_bias;
    
    SeqFullConnLayer(int input_dim, int output_dim) {
        _type = SEQ_FULL;
        _input_dim = input_dim;
        _output_dim = output_dim;

        _seq_full_weights.resize(_input_dim, _output_dim);
        _seq_full_bias.resize(1, _output_dim);

        _seq_full_weights.assign_val();
        _seq_full_bias.assign_val();

    }

    void _forward(Layer* pre_layer) {
        std::vector<matrix_double>().swap(_data);

        if (pre_layer->_type != RNN_CELL && pre_layer->_type != LSTM_CELL) {
            std::cerr << "Layer before seq faull conn layer is not rnn cell!" << std::endl;
            exit(1);
        }
        _seq_len = pre_layer->_seq_len;

        for (int i = 0; i < _seq_len; i++) {
            matrix_double value = pre_layer->_data[i] 
                * _seq_full_weights + _seq_full_bias;
            _data.push_back(value);
        }
        _pre_layer = pre_layer;

    }

    void _backward(Layer* nxt_layer) {
        
        if (nxt_layer->_type != SEQ_ACT) {
            std::cerr << "nxt nxt_layer is not seq loss nxt_layer!" << std::endl;
            exit(1);
        }
        std::vector<matrix_double>().swap(_errors);
        _delta_seq_full_weights.resize(_input_dim, _output_dim);
        _delta_seq_full_bias.resize(1, _output_dim);
        _errors = nxt_layer->_errors;
        for (int i = _seq_len - 1; i >= 0; i--) {
            _delta_seq_full_weights.add(_pre_layer->_data[i]._T() * _errors[i]);
            _delta_seq_full_bias.add(_errors[i]);
        }

    }

    void display() {
        std::cout << "------------full conn layer ----------" << std::endl;
        _seq_full_weights._display("_seq_full_weights");
        _seq_full_bias._display("_seq_full_bias");
        for (int i = 0; i < _seq_len; i++ ){
            _data[i]._display("data");
        }
        for (int i = 0; i < _seq_len; i++) {
            _errors[i]._display("error");
        }
        
    }

    void _update_gradient(int opt_type, double learning_rate) {
        if (opt_type == SGD) {
            _seq_full_weights.add(_delta_seq_full_weights * learning_rate);
            _seq_full_bias.add(_delta_seq_full_bias * learning_rate);
        }
    }


};

}
#endif
