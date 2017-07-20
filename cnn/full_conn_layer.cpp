#include "full_conn_layer.h"

namespace sub_dl {

void FullConnLayer::_forward(Layer* _pre_layer) {

	std::vector<matrix_double>().swap(_data);
	
	if (_pre_layer->_type == CONV ||
		_pre_layer->_type == POOL) {
		_pre_layer_data = _flatten(_pre_layer->_data);		
	} else {
		_pre_layer_data = _pre_layer->_data[0];
	}
	_data.push_back(sigmoid_m(_pre_layer_data
		* _full_conn_weights 
		+ _full_conn_bias));

}

void FullConnLayer::_backward(Layer* nxt_layer) {
	
	if (nxt_layer->_type == CONV || nxt_layer->_type == POOL) {
		std::cerr << "Conv or Pooling before full-connected not supported yet!" << std::endl;
		exit(1);
	}
	std::vector<matrix_double>().swap(_errors);
	matrix_double error;
	_nxt_layer = nxt_layer;

	if (nxt_layer->_type == LOSS) {
		error = nxt_layer->_errors[0].dot_mul(sigmoid_m_diff(_data[0]));
	} else if (nxt_layer->_type == FULL_CONN) {
		BaseFullConnLayer* full_conn_layer = (BaseFullConnLayer*) nxt_layer;
		error = (full_conn_layer->_errors[0]
			* full_conn_layer->_full_conn_weights._T())
			.dot_mul(sigmoid_m_diff(_data[0]));
	}

	_delta_full_conn_weights = _pre_layer_data._T() * error;
	_delta_full_conn_bias = error;
	_errors.push_back(error);	
}
}
