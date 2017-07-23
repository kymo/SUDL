#include "full_conn_layer.h"

namespace sub_dl {

void FullConnLayer::_forward(Layer* _pre_layer) {
    if (_pre_layer->_data.size() == 0 ||
        _pre_layer->_data[0]._x_dim > 1) {
        std::cerr << "Wrong Layer type before fullconnect layer!" << std::endl;
        exit(1);
    }

    std::vector<matrix_double>().swap(_data);
    _pre_layer_data = _pre_layer->_data[0];
    _data.push_back(_pre_layer_data
        * _full_conn_weights
        + _full_conn_bias);
	_raw_data.push_back(_data[0]);
}

void FullConnLayer::_backward(Layer* nxt_layer) {    
    if (nxt_layer->_type == CONV || nxt_layer->_type == POOL) {
        std::cerr << "Conv or Pooling before full-connected not supported yet!" << std::endl;
        exit(1);
    }
    std::vector<matrix_double>().swap(_errors);
    matrix_double error;
    _errors = nxt_layer->_errors;
    _delta_full_conn_weights = _pre_layer_data._T() * _errors[0];
    _delta_full_conn_bias = _errors[0];
}
}
