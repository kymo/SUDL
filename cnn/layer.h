#ifndef LAYER_H_
#define LAYER_H_

#include <iostream>
#include <map>
#include <vector>
#include "matrix.h"
#include "util.h"

namespace sub_dl {

enum {
    CONV = 0,
    POOL,
    FULLCONN,
	INPUT,
	LOSS,
};

class Layer {


public:
    
	std::vector<matrix_double> _data;
    std::vector<matrix_double> _errors;
    std::vector<matrix_double> _weights;
    int _type;
    
    int _feature_x_dim;
    int _feature_y_dim;

	Layer* _pre_layer;
	Layer* _nxt_layer;

    virtual void _forward(Layer* pre_layer) = 0;
    virtual void _backward(Layer* nxt_layer) = 0;

};

/*
class LossLayer : public Layer {
    
private:
    int _intput_dim;
    int _output_dim;
    matrix_double label;
    matrix_double _hidden_output_weights;
    matrix_double _output_bias;
    matrix_double _delta_hidden_output_weights;
    matrix_double _delta_output_bias;

public:
    virtual ~LossLayer() {}
    LossLayer(int input_dim, int output_dim) :
        _input_dim(input_dim), _output_dim(output_dim) {
        _hidden_output_weights.resize(input_dim, output_dim);
        _output_bias.resize(1, output_dim);
        _hidden_output_weights.assign_val();
        _output_bias.assign_val();
    }

    void _forward(Layer* _pre_layer) {
        _data.push_back(sigmoid_m(_pre_layer->_data[0] 
            * _hidden_output_weights 
            + _output_bias));
    }

    void _backward(Layer* _nxt_layer) {
        _errors.push_back((_data[0] - label)
            .dot_mul(sigmoid_m_diff(_data[0])));
		_delta_hidden_output_weights = _data[0]._T() * _errors[0];
		_delta_output_bias = _errors[0];
	}
};
*/

}
#endif
