
#ifndef CNN_H_
#define CNN_H_
#include <cstddef>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>
#include <string.h>
#include <fstream>
#include <map>
#include "matrix.h"
#include "loss_layer.h"
#include <iostream>

using namespace std;

namespace sub_ml {

class DataFeedLayer: public Layer {
public:
    DataFeedLayer(const std::vector<matrix_double>& data) {
        _data = data;
    }

    void _forward(Layer* pre_layer) {}
    void _backward(Layer* nxt_laery) {}

};

class CNN {

public:

    std::vector<std::vector<matrix_double> > train_x_feature;
    std::vector<matrix_double> train_y_label;

    std::vector<Layer*> _layers;

    void build_cnn(std::vector<Layer*> layers) {
        _layers = layers;
    }

    void _forward(std::vector<matrix_double> data) {
        Layer* pre_layer = new DataFeedLayer(data);
        for (auto layer : layers) {
            layer->_forward(pre_layer);
            pre_layer = layer;
            // data = layer->_forward(data);
        }
    }

    void _backward(const matrix_double& label) {
        //Layer* nxt_layer = new MeanSquareLossLayer(label);
        MeanSquareLossLayer* nxt_layer = (MeanSquareLossLayer*)_layers.back();
        nxt_layer->_set_data(label);
        for (int j = _layers.size() - 2; j >= 0; j--) {
            _layers[j]->_backward(nxt_layer);
            nxt_layer = _layers[i];
        }
    }

    void _update_gradient() {
        for (auto layer : layers) {
            layer->_update_gradient();
        }
    }

    void load_data(const char*file_name) {
        std::ifstream fis(file_name);
        int id, r, g, b, label;
        char comma;
        while (fis) {
            fis >> label;
            matrix_double label_data(1, 10);
            label_data[0][label - 1] = 1;
            vector<matrix_double> x_feature;
            matrix_double value(28, 28);
            for (int i = 0; i < 28; i++) {
                for (int j = 0; j < 28; j++) {
                    int v;
                    fis >> comma;
                    fis >> v;
                    std::cout << v << " ";
                    value[i][j] = comma;
                }
                std::cout << std::endl;
            }
            x_feature.push_back(value);
            train_x_feature.push_back(x_feature);
            train_y_label.push_back(label_data);
        }
        fis.close();
    }

    void train() {
        int epoch_cnt = 10;
        for (int epoch = 0; epoch < epoch_cnt; epoch++) {
            std::vector<std::vector<matrix_double> > batch_x_feature;
            std::vector<matrix_double> batch_y_label;
            for (int i = epoch * batch_size; i < 
                epoch * batch_size + batch_size; i++) {
                batch_x_feature.push_back(train_x_feature[i]);
                batch_y_label.push_back(train_y_label);
            }
            for (int i = 0; i < batch_x_feature.size(); i++) {
                _forward(batch_x_feature[i]);
                _backward(batch_y_label[i]);
                _update_gradient();
            }
        }
    }
};

}
#endif

