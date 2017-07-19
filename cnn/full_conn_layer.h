#ifndef FULLCONN_LAYER_H
#define FULLCONN_LAYER_H

#include "layer.h"
#include "base_full_conn_layer.h"

namespace sub_dl {

class FullConnLayer : public BaseFullConnLayer {

public:
    
	virtual ~FullConnLayer() {}
    
	FullConnLayer() {}
    
	void _forward(Layer* pre_layer);
	void _backward(Layer* nxt_layer);

};

}
#endif
