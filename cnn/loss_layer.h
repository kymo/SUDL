#ifndef LOSS_LAYER_H
#define LOSS_LAYER_H

#include "layer"

namespace sub_dl {

class LossLayer : public Layer {

public:
    matrix_double _label;
    LossLayer(const matrix_double& label) {
        _label = layer;
        _type = LOSS:
    }

};

class MeanSquareLossLayer : public LossLayer {

public:
    
	void _backward(Layer* nxt_layer) {
        _errors.push_back(pre_layer->_data[0] - _label);
    }
	
	void _forward(Layer* pre_layer) {
        if (pre_layer->_type != FULLCONN) {
            std::cerr << "Error pre layer for mean square loss layer" << std::endl;
            exit(1);
        }
        matrix_double minus_result = pre_layer->_data[0] - _label; 
        _data.push_back(minus_result.dot_mul(minus_result) * 0.5);
    }

};

class CrossEntropyLossLayer : public LossLayer {

public:
	void _forward(Layer* pre_layer) {
		if (pre_layer->_type != FULLCONN) {
			std::cerr << "Error pre layer for mean square loss layer" << std::endl;
			exit(1);
		}

		_data.push_back(label.dot_mul(log_m(pre_layer->_data[0])) 
			+ label.minus_by(1).dot_mul(log_m(vpre_layer->_data[0].minus_by(1))));

	}

	void _backward(Layer* nxt_layer) {

	}
};

#endif
