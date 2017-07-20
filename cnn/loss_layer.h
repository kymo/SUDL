#ifndef LOSS_LAYER_H
#define LOSS_LAYER_H

#include "layer.h"

namespace sub_dl {

class LossLayer : public Layer {

public:
    matrix_double _label;
    LossLayer() {
        _type = LOSS;
    }

	void _set_data(const matrix_double& label) {
		_label = label;
	}

	void _update_gradient(int opt_type, double learning_rate) {
	}

};

class MeanSquareLossLayer : public LossLayer {

public:

	MeanSquareLossLayer() {

	}
	
    void _backward(Layer* nxt_layer) {
		std::vector<matrix_double>().swap(_errors);
        _errors.push_back(_pre_layer->_data[0] - _label);
    }
    
    void _forward(Layer* pre_layer) {
		std::vector<matrix_double>().swap(_data);
		_pre_layer = pre_layer;
        if (pre_layer->_type != FULL_CONN) {
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
		std::vector<matrix_double>().swap(_data);
		_pre_layer = pre_layer;
        if (pre_layer->_type != FULL_CONN) {
            std::cerr << "Error pre layer for mean square loss layer" << std::endl;
            exit(1);
        }

        _data.push_back(_label.dot_mul(log_m(pre_layer->_data[0])) 
            + _label.minus_by(1).dot_mul(log_m(pre_layer->_data[0].minus_by(1))));

    }

    void _backward(Layer* nxt_layer) {

    }
};

}

#endif
