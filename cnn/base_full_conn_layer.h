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
	matrix_double _pre_layer_data;
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

	void display() {
		std::cout << "--------------full conn layer-----------" << std::endl;
		_data[0]._display("data");
		if (_errors.size() > 0) 
		_errors[0]._display("error");
		std::cout << "--------------full conn layer end-----------" << std::endl;
	}

	matrix_double _flatten(const std::vector<matrix_double>& data) {
		
		if (data.size() == 0) {
			std::cerr << "Flatten error!" << std::endl;
			exit(1);
		}
		int data_size = data.size();
		int data_x_dim = data[0]._x_dim;
		int data_y_dim = data[0]._y_dim;
		matrix_double t_matrix(1, data_size * data_x_dim * data_y_dim);
		for (int i = 0; i < t_matrix._y_dim; i++) {
			int n = i / (data_x_dim * data_y_dim);
			int inner_n = i - n * (data_x_dim * data_y_dim);
			t_matrix[0][i] = data[n][inner_n / data_y_dim][inner_n % data_y_dim]; 
		}
		return t_matrix;

	}
	
	void _update_gradient(int opt_type, double learning_rate) {
		if (opt_type == SGD) {
			_full_conn_weights.add(_delta_full_conn_weights * learning_rate);
			_full_conn_bias.add(_delta_full_conn_bias * learning_rate);
		}
	}

};

}
#endif
