#ifndef ACTIVE_LAYER_H
#define ACTIVE_LAYER_H
#include "layer.h"
#include "flat_layer.h"
#include "full_conn_layer.h"
#include "active_func.h"

namespace sub_dl {

class ActiveLayer : public Layer {

public:
    ActiveFunc<double>* _active_func;
    
    ~ActiveLayer() {}
    ActiveLayer() {
        _type = ACT;
    }

    void _forward(Layer* pre_layer) {
        std::vector<matrix_double>().swap(_data);
        if (pre_layer->_type == CONV ||
            pre_layer->_type == POOL) {
            for (int i = 0; i < pre_layer->_output_dim; i++) {
                _data.push_back(_active_func->_calc(pre_layer->_data[i]));
            }
        } else if (pre_layer->_type == FULL_CONN) {
            _data.push_back(_active_func->_calc(pre_layer->_data[0]));
        } else {
            std::cerr << "Error when " << std::endl;
            exit(1);
        }
        _output_dim = pre_layer->_output_dim;
        _feature_x_dim = pre_layer->_feature_x_dim;
        _feature_y_dim = pre_layer->_feature_y_dim;
    }

    void _backward(Layer* nxt_layer) {
        std::vector<matrix_double>().swap(_errors);
        matrix_double error;
        if (nxt_layer->_type == LOSS) {
            error = nxt_layer->_errors[0]
                .dot_mul(_active_func->_diff(_pre_layer->_data[0]));
            _errors.push_back(error);
        } else if (nxt_layer->_type == FULL_CONN) {
            BaseFullConnLayer* full_conn_layer = (BaseFullConnLayer*) nxt_layer;
            error = (full_conn_layer->_errors[0]
                * full_conn_layer->_full_conn_weights._T())
                .dot_mul(_active_func->_diff(_pre_layer->_data[0]));
            _errors.push_back(error);

        } else if (nxt_layer->_type == FLAT) {
            FlatternLayer* flat_layer = (FlatternLayer*) nxt_layer;
            for (int i = 0; i < _output_dim; i++) {
                matrix_double error(_feature_x_dim, _feature_y_dim);
                for (int u = 0; u < error._x_dim; u++) {
                    for (int v = 0; v < error._y_dim; v++) {
                        error[u][v] = flat_layer->_errors[0][0][i * (_feature_x_dim * _feature_y_dim)
                            + u * _feature_y_dim + v];
                    }
                }
                error = error.dot_mul(_active_func->_diff(_pre_layer->_data[i]));
                _errors.push_back(error);
            }
        } else if (nxt_layer->_type == CONV) {
            const ConvLayer* conv_layer = (ConvLayer*)(nxt_layer);
            for (int i = 0; i < _output_dim; i++) {
                matrix_double error(_feature_x_dim, _feature_y_dim);
                for (int j = 0; j < conv_layer->_output_dim; j++) {
                    if (conv_layer->_conn_map[i][j]) {
                        matrix_double conv2d_vec = conv_layer->_errors[j]
                            .conv2d(conv_layer->_conv_kernels[i][j].rotate_180(), FULL);
                        error = error + conv2d_vec;
                    }
                }
                error = error.dot_mul(_active_func->_diff(_pre_layer->_data[i]));
                _errors.push_back(error);
            }
        } else if (nxt_layer->_type == POOL) {
            const PoolingLayer* pooling_layer = (PoolingLayer*)(nxt_layer);
            int delta_size = (pooling_layer->_pooling_x_dim) * pooling_layer->_pooling_y_dim;
            for (int i = 0; i < _output_dim; i++) {
                matrix_double up_delta = pooling_layer->_errors[i]
                    .up_sample(pooling_layer->_pooling_x_dim, 
                    pooling_layer->_pooling_y_dim);
                matrix_double error = up_delta
                    .dot_mul(_active_func->_diff(_pre_layer->_data[i]))
                    * (pooling_layer->_pooling_weights[0][i] / delta_size);
                _errors.push_back(error);
            }
        }
    }

    void _update_gradient(int opt_type, double learning_rate) {
    }

    void display() {
    }    

};

class SigmoidLayer : public ActiveLayer {
public:
    SigmoidLayer() {
        _active_func = ActiveFuncFactory<double>::_get_instance()
            ->_produce(SIGMOID);
        if (NULL == _active_func) {
            exit(1);
        }
    }
};

class ReluLayer : public ActiveLayer {
public:
    ReluLayer() {
        _active_func = ActiveFuncFactory<double>::_get_instance()
            ->_produce(RELU);
        if (NULL == _active_func) {
            exit(1);
        }
    }
};

}
#endif
