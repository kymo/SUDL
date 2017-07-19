#ifndef POOLING_LAYER_H
#define POOLING_LAYER_H

#include "layer.h"

namespace sub_dl {

class PoolingLayer : public Layer {

private:
    int _input_dim;
    int _output_dim;

public:
    int _pooling_x_dim;
    int _pooling_y_dim;
    int _pooling_type; // 0 max_pooling 1 avg_pooling
    matrix_double _pooling_weights;
    matrix_double _pooling_bias;
    matrix_double _delta_pooling_weights;
    matrix_double _delta_pooling_bias;
    virtual ~ PoolingLayer() {}
    
    PoolingLayer() {}

    PoolingLayer(int input_dim, int output_dim, 
        int pooling_x_dim, int pooling_y_dim, 
        int feature_x_dim, int feature_y_dim);

    void display();
    
    void _forward(Layer* pre_layer);
    
    void _backward(Layer* _nxt_layer);

	void _update_gradient(int opt_type, double learning_rate);
};

}
#endif
