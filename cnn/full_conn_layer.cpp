#include "full_conn_layer.h"

namespace sub_dl {

void FullConnLayer::_forward(Layer* _pre_layer) {

	matrix_double input_data;
	if (_pre_layer->_type == CONV || 
		_pre_layer->_type == POOL) {
		input_data = _flatten(_pre_layer->_data);
	} else {
		input_data = _pre_layer->_data[0];
	}

	_data.push_back(sigmoid_m(input_data 
		* _full_conn_weights 
		+ _full_conn_bias));

}

void FullConnLayer::_backward(Layer* nxt_layer) {
	
	if (nxt_layer->_type == CONV || nxt_layer->_type == POOL) {
		std::cerr << "Conv or Pooling before full-connected not supported yet!" << std::endl;
		exit(1);
	}
	matrix_double error;
	if (nxt_layer->_type == LOSS) {
		error = nxt_layer->_errors[0].dot_mul(sigmoid_m_diff(_data[0]));
	} else if (nxt_layer->_type == FULL) {
		BaseFullConnLayer* full_conn_layer = (BaseFullConnLayer*) nxt_layer;
		error = full_conn_layer->_errors[0] 
			* full_conn_layer->_full_conn_weights
			.dot_mul(sigmoid_m_diff(_data[0]));
	}

	_errors.push_back(error);	
}
}
