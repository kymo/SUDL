#ifndef FULLCONN_LAYER_H
#define FULLCONN_LAYER_H

#include "layer.h"
#include "base_full_conn_layer.h"

namespace sub_dl {

class FullConnLayer : public BaseFullConnLayer {

public:
    
	virtual ~FullConnLayer() {}

	FullConnLayer(int input_dim, int output_dim) :
		BaseFullConnLayer(input_dim, output_dim) {}

	void _forward(Layer* pre_layer);
	void _backward(Layer* nxt_layer);

	void _update_gradient(int opt_type, double learning_rate) {
		if (opt_type == SGD) {
			_full_conn_weights.add(_delta_full_conn_weights * learning_rate);
			_full_conn_bias.add(_delta_full_conn_bias * learning_rate);
		}
	}
};

}
#endif
