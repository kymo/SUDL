// conv_layer.cpp

#include "conv_layer.h"
#include "pooling_layer.h"

namespace sub_dl {


ConvLayer::ConvLayer(int input_dim, int output_dim, int kernel_x_dim, int kernel_y_dim, int feature_x_dim, int feature_y_dim) {

    _input_dim = input_dim;
    _output_dim = output_dim;
    _kernel_x_dim = kernel_x_dim;
    _kernel_y_dim = kernel_y_dim;
    _feature_x_dim = feature_x_dim;
    _feature_y_dim = feature_y_dim;
    _conv_kernels.resize(_input_dim, _output_dim);
    _feature_x_dim = feature_x_dim;
    _feature_y_dim = feature_y_dim;

    for (int i = 0; i < _input_dim;i ++) {
        for (int j = 0; j < _output_dim; j++) {    
            matrix_double kernel(_kernel_x_dim, _kernel_y_dim);
            kernel.assign_val();
            _conv_kernels[i][j] = kernel;
        }
    }
    _bias.resize(1, _output_dim);
    _bias.assign_val();
    _type = CONV;

}

void ConvLayer::display() {
    
    std::cout << "-----------feature map------" << std::endl;
    for (int i = 0; i < _output_dim;i ++) {
        std::cout << "feature map " << i << std::endl;
        _data[i]._display();
    }
    std::cout << "-----------kernel------" << std::endl;
    for (int i = 0; i < _input_dim;i ++) {
        for (int j = 0; j < _output_dim;j ++) {
            std::cout << "kernel " << i << " " << j << std::endl;
			if (_conn_map[i][j]) {
            	_conv_kernels[i][j]._display();
			} else {
				std::cout << "NULL" << std::endl;
			}
        }
    }
    std::cout << "-----------bias-------" << std::endl;
    _bias._display();
}

void ConvLayer::_forward(Layer* _pre_layer) {
    for (int i = 0; i < _output_dim; i++) {
        matrix_double feature_map(_feature_x_dim, _feature_y_dim);
        for (int j = 0; j < _input_dim; j++) {
            if (_conn_map[j][i]) {
                feature_map = feature_map + _pre_layer->_data[j].conv(_conv_kernels[j][i]);
            }
        }
		feature_map._display("-featuremap-");
        _data.push_back(sigmoid_m(feature_map + _bias[0][i]));
    }
}

void ConvLayer::_backward(Layer* _nxt_layer) {
    if (_nxt_layer->_type == POOL) {
        const PoolingLayer* pooling_layer = (PoolingLayer*) _nxt_layer;
        for (int i = 0; i < _output_dim; i++) {
            matrix_double t1 = pooling_layer->_errors[i]
                .up_sample(pooling_layer->_pooling_x_dim, 
                pooling_layer->_pooling_y_dim);
            t1._display("t1");
            matrix_double error = t1.dot_mul(sigmoid_m_diff(_data[i]))
                * pooling_layer->_pooling_weights[0][i];
            std::cout << "pooling weights:" << pooling_layer->_pooling_weights[0][i] << std::endl;
            error._display("error");
            _errors.push_back(error);
        }
    }
}

}
