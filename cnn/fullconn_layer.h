#ifndef FULLCONN_LAYER_H
#define FULLCONN_LAYER_H

#include "layer.h"

namespace sub_dl {

class FullConnLayer : public Layer {

private:
    int _input_dim;
    int _output_dim;

public:
    matrix_double _full_conn_weights;
    matrix_double _full_conn_bias;
    matrix_double _delta_full_conn_weights;
    matrix_double _delta_full_conn_bias;

    virtual ~FullConnLayer() {}
    FullConnLayer() {}
    FullConnLayer(int intput_dim, int output_dim) {
        _full_conn_weights.resize(input_dim, output_dim);
        _full_conn_bias.resize(1, output_dim);
        _full_conn_weights.assign_val();
        _full_conn_bias.assign_val();
		_type = FULLCONN;
    }

    void _forward(Layer* _pre_layer) {
        _input_data = Matrix<double>::concat(_pre_layer->_data);
        _data.push_back(sigmoid_m(_input_data 
            * _full_conn_weights 
            + _full_conn_weights));
    }
    void _backward(Layer* _nxt_layer) {
		matrix_double error = _nxt_layer._erros[0] 
            * _nxt_layer._weights[0]._T()
            .dot_mul(sigmoid_m_diff(_data[0]));
        _errors.push_back(error);	
	}
};

}
#endif
