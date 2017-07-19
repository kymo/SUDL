#include "pooling_layer.h"
#include "conv_layer.h"

namespace sub_dl {

PoolingLayer::PoolingLayer(int input_dim, int output_dim, int pooling_x_dim, int pooling_y_dim,
    int feature_x_dim, int feature_y_dim) {

    _input_dim = input_dim;
    _output_dim = output_dim;
    _pooling_x_dim = pooling_x_dim;
    _pooling_y_dim = pooling_y_dim;
    _feature_x_dim = feature_x_dim;
    _feature_y_dim = feature_y_dim;
    _pooling_weights.resize(1, output_dim);
    _pooling_weights.assign_val();
    _pooling_bias.resize(1, output_dim);
    _pooling_bias.assign_val();
    _delta_pooling_weights.resize(1, output_dim);
    _delta_pooling_bias.resize(1, output_dim);
    _type = POOL;

}

void PoolingLayer::display() {
    _pooling_weights._display("_pooling_weights");
    _pooling_bias._display("_pooling_bias");
    std::cout << "-----------pooling output --------" << std::endl;
    for (int i = 0; i < _output_dim; i++) {
        _data[i]._display();
    }
}

void PoolingLayer::_forward(Layer* pre_layer) {
    _pre_layer = pre_layer;
    for (int i = 0; i < _output_dim; i++) {
        matrix_double up_feature = pre_layer->_data[i]
            .down_sample(_pooling_x_dim, _pooling_y_dim, AVG_POOLING) * _pooling_weights[0][i] 
            + _pooling_bias[0][i];
        _data.push_back(sigmoid_m(up_feature));
    }
}

void PoolingLayer::_backward(Layer* nxt_layer) {
    _nxt_layer = nxt_layer;
    const ConvLayer* conv_layer = (ConvLayer*)(nxt_layer);
    for (int i = 0; i < _output_dim; i++) {
        matrix_double error(_feature_x_dim, _feature_y_dim);
        for (int j = 0; j < conv_layer->_output_dim; j++) {
            if (conv_layer->_conn_map[j][i]) {
                conv_layer->_errors[i]._display("conv_layer->_errors[i]");
                conv_layer->_conv_kernels[i][j]._display("conv_layer->_conv_kernels[i][j]");
                matrix_double conv2d_vec = conv_layer->_errors[i].conv2d(conv_layer->_conv_kernels[i][j], FULL);
                conv2d_vec._display("conv2d_vec");
                sigmoid_m_diff(_data[i])._display("sigmoid_m_diff(_data[i])");
                error = error + conv2d_vec.dot_mul(sigmoid_m_diff(_data[i]));
            }
        }
        _errors.push_back(error);
        _delta_pooling_weights[0][i] = (error
            .dot_mul(_pre_layer->_data[i].down_sample(_pooling_x_dim, _pooling_y_dim, AVG_POOLING)))
            .sum();
        _delta_pooling_bias[0][i] = error.sum();
    }
}

}
