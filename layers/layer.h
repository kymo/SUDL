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
	FLAT,
	// RNN
	SEQ_FULL,
	SEQ_SOFTMAX,
	SEQ_ACT,
	SEQ_LOSS,
	RNN_CELL
} layer_type;

enum {
	SGD = 0,
} opt_type;

class Layer {

public:

	std::vector<matrix_double> _data;
    std::vector<matrix_double> _errors;
    
	// layer type
	int _type;
    
	// x_dim & y_dim of feature map
    int _feature_x_dim;
    int _feature_y_dim;
	
	// rnn: _input_dim & _outpupt_dim indicate the feature length 
	// and output length in time t
	// cnn: _input_dim & _output_dim indicate the number of input 
	// or output feature map
	int _input_dim;
    int _output_dim;

	// use only for sequence model
	int _seq_len;	

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

    DataFeedLayer (const std::vector<matrix_double>& data) {
		_type = INPUT;
		_data = data;
    }
	~DataFeedLayer() {
	}

	void _set_data(const std::vector<matrix_double>& data) {
        _data = data;
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
