#ifndef CNN_H_
#define CNN_H_

#include <fstream>
#include "matrix.h"
#include "loss_layer.h"

namespace sub_dl {

template <typename T>
class CNN {

public:

    std::vector<std::vector<matrix_double> > train_x_feature;
    std::vector<matrix_double> train_y_label;

    std::vector<Layer*> _layers;

    void build_cnn(std::vector<Layer*> layers) {
        _layers = layers;
    }

    void _forward(std::vector<matrix_double> data) {
        Layer* data_layer = new DataFeedLayer(data);
		Layer* pre_layer = data_layer;
        for (auto layer : _layers) {
            layer->_forward(pre_layer);
            pre_layer = layer;
        }
		/*
		if (NULL != data_layer) {
			delete data_layer;
		}*/
	}



    double _backward(const matrix_double& label) {
        Layer* loss_layer = new T(label);
        loss_layer->_forward(_layers.back());
        loss_layer->_backward(NULL);
      	
		Layer* nxt_layer = loss_layer;
		double cost = 0.0;
        cost += nxt_layer->_data[0].sum();
        for (int j = _layers.size() - 1; j >= 0; j--) {
            _layers[j]->_backward(nxt_layer);
            nxt_layer = _layers[j];
        }
		/*
		if (NULL != loss_layer) {
			delete loss_layer;
		}*/
        return cost;
    }

    void _update_gradient() {
        for (auto layer : _layers) {
            layer->_update_gradient(SGD, -0.01);
        }
    }
    
    void load_test_data(const char*file_name) {
        matrix_double label_data(1, 3);
        label_data[0][1] = 1;
        std::vector<matrix_double> x_feature;
        matrix_double value(6, 6);
        // std::cout << label << std::endl;
        for (int i = 0; i < 36; i++) {
            value[i / 6][i % 6] = rand() % 4;
        }
        x_feature.push_back(value);
        train_x_feature.push_back(x_feature);
        train_y_label.push_back(label_data);
    }
    
    void load_iris_data(const char*file_name) {
        std::ifstream fis(file_name);
        int id, r, g, b, label;
        char comma;
        while (fis >> label) {
            matrix_double label_data(1, 3);
            label_data[0][label] = 1;
            std::vector<matrix_double> x_feature;
            matrix_double value(2, 2);
            // std::cout << label << std::endl;
            for (int i = 0; i < 4; i++) {
                double v;
                fis >> comma;
                fis >> v;
                value[i / 2][i % 2] = v * 1.0;
            }
            x_feature.push_back(value);
            train_x_feature.push_back(x_feature);
            train_y_label.push_back(label_data);
        }
        for (int i = 0; i < train_x_feature.size(); i++) {
            int v = rand() % (train_x_feature.size());
            int s = rand() % (train_x_feature.size());
            //std::swap(train_x_feature[v], train_x_feature[s]);
            //std::swap(train_y_label[v], train_y_label[s]);
        }
        // fis.close();
        std::cout << "Load Data " << train_x_feature.size() << std::endl;
    }

    void load_data(const char*file_name) {
        std::ifstream fis(file_name);
        int id, r, g, b, label;
        char comma;
        while (fis >> label) {
            matrix_double label_data(1, 4);
			int i = 0;
			while (label > 0) {
				label_data[0][i ++] = label % 2;
				label /= 2;
			}
            // label_data[0][label] = 1;
            std::vector<matrix_double> x_feature;
            matrix_double value(28, 28);
            // std::cout << label << std::endl;
            for (int i = 0; i < 28; i++) {
                for (int j = 0; j < 28; j++) {
                    int v;
                    fis >> comma;
                    fis >> v;
                    value[i][j] = v / 256.0;
                }
                //std::cout << std::endl;
            }
            x_feature.push_back(value);
            train_x_feature.push_back(x_feature);
            train_y_label.push_back(label_data);
        }
        
        // fis.close();
        std::cout << "Load Data" << std::endl;
    }
    
    int merge(const matrix_double& data) {
        double v = -100;
        int id = 0;
        
		for (int i = data._y_dim - 1; i >= 0; i--) {
			if (data[0][i] > 0.5) {
				id += 1 << (data._y_dim - 1 - i);
			}
			/*
            if (v < data[0][i]) {
                v = data[0][i];
                id = i;
            }*/
        }
        return id;
    }

    void gradient_check(std::vector<matrix_double> batch_x_feature,
        matrix_double label) {   
        for (int l = 0; l < _layers.size(); l++) {
            if (_layers[l]->_type == FULL_CONN) {
                BaseFullConnLayer* layer = (BaseFullConnLayer*)_layers[l];
                std::cout << "--------Full Gradient Check Result----------" << std::endl;
                for (int i = 0; i < layer->_input_dim; i++) {
                    for (int j = 0; j < layer->_output_dim; j++) {
                        double v = layer->_full_conn_weights[i][j];
                        layer->_full_conn_weights[i][j] = v + 1.0e-4;
                        _forward(batch_x_feature);
                        Layer* nxt_layer = new T(label);
                        nxt_layer->_forward(_layers.back());
                        nxt_layer->_backward(NULL);
                        double f1 = nxt_layer->_data[0].sum();    
                        layer->_full_conn_weights[i][j] = v - 1.0e-04;
                        _forward(batch_x_feature);
                        
                        nxt_layer = new T(label);
                        nxt_layer->_forward(_layers.back());
                        nxt_layer->_backward(NULL);
                        double f2 = nxt_layer->_data[0].sum();    
                        std::cout << "[ " << layer->_delta_full_conn_weights[i][j] << " " << (f1 - f2) / (2.0e-4) << "], ";
                        layer->_full_conn_weights[i][j] = v;
                    }
                    std::cout << std::endl;
                }
                std::cout << "bias:" << std::endl;
                for (int j = 0; j < layer->_output_dim;j ++) {
                    double v = layer->_full_conn_bias[0][j];
                    layer->_full_conn_bias[0][j] = v + 1.0e-4;
                    _forward(batch_x_feature);
                    Layer* nxt_layer = new T(label);
                    nxt_layer->_forward(_layers.back());
                    nxt_layer->_backward(NULL);
                    double f1 = nxt_layer->_data[0].sum();    
                    layer->_full_conn_bias[0][j] = v - 1.0e-04;
                    _forward(batch_x_feature);
                    
                    nxt_layer = new T(label);
                    nxt_layer->_forward(_layers.back());
                    nxt_layer->_backward(NULL);
                    double f2 = nxt_layer->_data[0].sum();    
                    std::cout << "[ " << layer->_delta_full_conn_bias[0][j] << " " << (f1 - f2) / (2.0e-4) << "], ";
                    layer->_full_conn_bias[0][j] = v;
                }
                std::cout << std::endl;
            } else if (_layers[l]->_type == CONV) {
                ConvLayer* layer = (ConvLayer*) _layers[l];
                std::cout << "-------Conv Layer Gradient Check Result ---------" << std::endl;
                for (int i = 0; i < layer->_input_dim;i ++) {
                    for (int j = 0; j < layer->_output_dim; j++) {
                        for (int u = 0; u < layer->_kernel_x_dim; u++) {
                            for (int v = 0; v < layer->_kernel_y_dim; v++) {
                                double val = layer->_conv_kernels[i][j][u][v];
                                layer->_conv_kernels[i][j][u][v] = val + 1.0e-4;
                                _forward(batch_x_feature);
                                Layer* nxt_layer = new T(label);
                                nxt_layer->_forward(_layers.back());
                                nxt_layer->_backward(NULL);
                                double f1 = nxt_layer->_data[0].sum();
                                layer->_conv_kernels[i][j][u][v] = val - 1.0e-4;
                                _forward(batch_x_feature);
                                
                                nxt_layer = new T(label);
                                nxt_layer->_forward(_layers.back());
                                nxt_layer->_backward(NULL);
                                double f2 = nxt_layer->_data[0].sum();

                                std::cout << "[ " << layer->_delta_conv_kernels[i][j][u][v] << " " << (f1 - f2) / (2.0e-4) << "], ";
                                layer->_conv_kernels[i][j][u][v] = val;
                            }
                            std::cout << std::endl;
                        }

                    }
                }
                std::cout << "bias:" << std::endl;
                int i = 0;
                for (int j = 0; j < layer->_output_dim; j++) {
                    std::cout << layer->_conv_bias._x_dim << " " << layer->_conv_bias._y_dim << std::endl;
                    double val = layer->_conv_bias[i][j];
                    layer->_conv_bias[i][j] = val + 1.0e-4;
                    _forward(batch_x_feature);
                    Layer* nxt_layer = new T(label);
                    nxt_layer->_forward(_layers.back());
                    nxt_layer->_backward(NULL);
                    double f1 = nxt_layer->_data[0].sum();    
                    layer->_conv_bias[i][j] = val - 1.0e-4;
                    _forward(batch_x_feature);
                    
                    nxt_layer = new T(label);
                    nxt_layer->_forward(_layers.back());
                    nxt_layer->_backward(NULL);
                    double f2 = nxt_layer->_data[0].sum();
                    std::cout << f1 - f2 << std::endl;
                    std::cout << "[ " << layer->_delta_conv_bias[i][j] << " " << (f1 - f2) / (2.0e-4) << "], ";
                    layer->_conv_bias[i][j] = val;
                }
                std::cout << std::endl;
            } else if (_layers[l]->_type == POOL) {
                PoolingLayer* layer = (PoolingLayer*) _layers[l];
                std::cout << "------Poolign Layer Gradient Check Result ---------" << std::endl;
                for (int j = 0; j < layer->_output_dim; j++) {
                    double v = layer->_pooling_weights[0][j];
                    layer->_pooling_weights[0][j] = v + 1.0e-4;
                    _forward(batch_x_feature);
                    Layer* nxt_layer = new T(label);
                    nxt_layer->_forward(_layers.back());
                    nxt_layer->_backward(NULL);
                    double f1 = nxt_layer->_data[0].sum();    
                    layer->_pooling_weights[0][j] = v - 1.0e-4;
                    _forward(batch_x_feature);
                    
                    nxt_layer = new T(label);
                    nxt_layer->_forward(_layers.back());
                    nxt_layer->_backward(NULL);
                    double f2 = nxt_layer->_data[0].sum();

                    std::cout << "[ " << layer->_delta_pooling_weights[0][j] << " " << (f1 - f2) / (2.0e-4) << "], ";
                    layer->_pooling_weights[0][j] = v;
                }
                std::cout << std::endl;
                std::cout << "bias" << std::endl;
                for (int j = 0; j < layer->_output_dim; j++) {
                    double v = layer->_pooling_bias[0][j];
                    layer->_pooling_bias[0][j] = v + 1.0e-4;
                    _forward(batch_x_feature);
                    Layer* nxt_layer = new T(label);
                    nxt_layer->_forward(_layers.back());
                    nxt_layer->_backward(NULL);
                    double f1 = nxt_layer->_data[0].sum();    
                    layer->_pooling_bias[0][j] = v - 1.0e-4;
                    _forward(batch_x_feature);
                    
                    nxt_layer = new T(label);
                    nxt_layer->_forward(_layers.back());
                    nxt_layer->_backward(NULL);
                    double f2 = nxt_layer->_data[0].sum();

                    std::cout << "[ " << layer->_delta_pooling_bias[0][j] << " " << (f1 - f2) / (2.0e-4) << "], ";
                    layer->_pooling_bias[0][j] = v;
                }
                std::cout << std::endl;
            }
        }
    }

    void train() {
        int epoch_cnt = 100;
        int batch_size = 10;
        int tot = 150;
        for (int epoch = 0; epoch < epoch_cnt; epoch++) {
            for (int batch = 0; batch < tot / batch_size; batch++) {
                std::vector<std::vector<matrix_double> > batch_x_feature;
                std::vector<matrix_double> batch_y_label;
                for (int i = batch * batch_size; i < 
                    batch * batch_size + batch_size; i++) {
                    batch_x_feature.push_back(train_x_feature[i]);
                    batch_y_label.push_back(train_y_label[i]);
                }
                double tot = 0.0;
                int v1, v2;
                for (int i = 0; i < batch_x_feature.size(); i++) {
                    _forward(batch_x_feature[i]);               
                    double cost = _backward(batch_y_label[i]);
                    v1 = merge(batch_y_label[i]);
                    v2 = merge(_layers.back()->_data[0]);
                    tot += cost;
                    // gradient check
                    // gradient_check( batch_x_feature[i], batch_y_label[i]);
                    _update_gradient();
                }
                std::cout << "Cost " << tot << " " << v1 << " - " << v2 << std::endl;
            }
        }
    }
};

}
#endif

