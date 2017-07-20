#ifndef CNN_H_
#define CNN_H_

#include <fstream>
#include "matrix.h"
#include "loss_layer.h"

namespace sub_dl {

class DataFeedLayer: public Layer {
public:
    DataFeedLayer(const std::vector<matrix_double>& data) {
        _data = data;
    }

    void _forward(Layer* pre_layer) {}
    void _backward(Layer* nxt_laery) {}

	void _update_gradient(int opt_type, double learning_rate) {}

};


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
        Layer* pre_layer = new DataFeedLayer(data);
        for (auto layer : _layers) {
            layer->_forward(pre_layer);
            pre_layer = layer;
            // data = layer->_forward(data);
        }
    }

    double _backward(const matrix_double& label) {
        // MeanSquareLossLayer* nxt_layer = new MeanSquareLossLayer(label);
		Layer* nxt_layer = new T();
		((T*)(nxt_layer))->_set_data(label);
		nxt_layer->_forward(_layers.back());
		nxt_layer->_backward(NULL);

		double cost = 0.0;
		cost += nxt_layer->_data[0].sum();
		//Layer* nxt_layer = _layers.back();
        //((MeanSquareLossLayer*)nxt_layer)->_set_data(label);
        for (int j = _layers.size() - 1; j >= 0; j--) {
            _layers[j]->_backward(nxt_layer);
            nxt_layer = _layers[j];
        }
		return cost;

    }

    void _update_gradient() {
        for (auto layer : _layers) {
            layer->_update_gradient(SGD, -0.05);
        }
    }

    void load_data(const char*file_name) {
        std::ifstream fis(file_name);
        int id, r, g, b, label;
        char comma;
        while (fis >> label) {
            matrix_double label_data(1, 10);
            label_data[0][label] = 1;
            std::vector<matrix_double> x_feature;
            matrix_double value(28, 28);
			// std::cout << label << std::endl;
            for (int i = 0; i < 28; i++) {
                for (int j = 0; j < 28; j++) {
                    int v;
                    fis >> comma;
                    fis >> v;
               //     std::cout << v << " ";
                    value[i][j] = v;
                }
                //std::cout << std::endl;
            }
            x_feature.push_back(value);
            train_x_feature.push_back(x_feature);
            train_y_label.push_back(label_data);
        	value._display("x_feature");
			label_data._display("lbael_data");
		}
		
        // fis.close();
		std::cout << "Load Data" << std::endl;
    }

    void train() {
        int epoch_cnt = 10;
		int batch_size = 10;
        for (int epoch = 0; epoch < epoch_cnt; epoch++) {
            std::vector<std::vector<matrix_double> > batch_x_feature;
            std::vector<matrix_double> batch_y_label;
            for (int i = epoch * batch_size; i < 
                epoch * batch_size + batch_size; i++) {
                batch_x_feature.push_back(train_x_feature[i]);
                batch_y_label.push_back(train_y_label[i]);
            }
            for (int i = 0; i < batch_x_feature.size(); i++) {
                _forward(batch_x_feature[i]);				
                double cost = _backward(batch_y_label[i]);
                _update_gradient();
				std::cout << cost << std::endl;
            }
        }
    }
};

}
#endif

