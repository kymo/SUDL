#ifndef BASE_FULLCONN_LAYER_H
#define BASE_FULLCONN_LAYER_H

#include "layer.h"

namespace sub_dl {

class BaseFullConnLayer : public Layer {

public:
    int _input_dim;
    int _output_dim;
    matrix_double _full_conn_weights;
    matrix_double _full_conn_bias;
    matrix_double _delta_full_conn_weights;
    matrix_double _delta_full_conn_bias;

	BaseFullConnLayer() {}
    
	BaseFullConnLayer(int input_dim, int output_dim) : _input_dim(input_dim),
		_output_dim(output_dim) {
        _full_conn_weights.resize(input_dim, output_dim);
        _full_conn_bias.resize(1, output_dim);
        _full_conn_weights.assign_val();
        _full_conn_bias.assign_val();
		_delta_full_conn_weights.resize(input_dim, output_dim);
		_delta_full_conn_bias.resize(1, output_dim);
		_type = FULL_CONN;
    }

	matrix_double _flatten(const std::vector<matrix_double>& data) {
		
		if (data.size() == 0) {
			std::cerr << "Flatten error!" << std::endl;
			exit(1);
		}
		matrix_double t_matrix(1, data.size() * data[0]._x_dim * data[0]._y_dim);
		for (int i = 0; i < t_matrix._y_dim; i++) {
			int n = i / (data[0]._x_dim * data[0]._y_dim);
			t_matrix[0][i] = data[n][i / data[0]._y_dim][i % data[0]._y_dim]; 
		}
		return t_matrix;

	}

	virtual void _forward(Layer* pre_layer) {};

	virtual void _backward(Layer* nxt_layer) {};


};

}
#endif
