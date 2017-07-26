#ifndef FULLCONN_SOFTMAX_LAYER_H
#define FULLCONN_SOFTMAX_LAYER_H

#include "layer.h"
#include "base_full_conn_layer.h"

namespace sub_dl {

class FullConnSoftmaxLayer : public BaseFullConnLayer {

public:
    virtual ~FullConnSoftmaxLayer() {}
    FullConnSoftmaxLayer() {}
	
	void _forward(Layer* pre_layer);
	void _backward(Layer* nxt_layer);

};

}
#endif
