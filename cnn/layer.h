#ifndef LAYER_H_
#define LAYER_H_

#include <iostream>
#include <map>
#include <vector>
#include "matrix.h"
#include "util.h"
#include "active_func.h"
namespace sub_dl {

enum {
    CONV = 0,
    POOL,
    FULL_CONN,
	INPUT,
	LOSS,
	ACT,
	FLAT
} layer_type;

enum {
	SGD = 0,
} opt_type;

class Layer {

public:

	std::vector<matrix_double> _data;
	std::vector<matrix_double> _raw_data;
    std::vector<matrix_double> _errors;
    
	int _type;
    
    int _feature_x_dim;
    int _feature_y_dim;
    
	int _input_dim;
    int _output_dim;

	Layer* _pre_layer;
	Layer* _nxt_layer;
	ActiveFunc<double>* _active_func;

    virtual void _forward(Layer* pre_layer) = 0;
    virtual void _backward(Layer* nxt_layer) = 0;
	virtual void _update_gradient(int opt_type, double learning_rate) = 0;
	virtual void display() = 0;
};

class DataFeedLayer: public Layer {
public:
    DataFeedLayer(const std::vector<matrix_double>& data) {
        _data = data;
		_type = INPUT;
    }

    void _forward(Layer* pre_layer) {}
    void _backward(Layer* nxt_laery) {}

	void _update_gradient(int opt_type, double learning_rate) {}
	void display() {
		std::cout << "-----------input layer----------" << std::endl;
		_data[0]._display("data");
	}
};


}
#endif
