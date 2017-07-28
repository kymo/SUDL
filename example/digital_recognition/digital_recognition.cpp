
#include <iostream>
#include <fstream>
#include "net_wrapper.h"

#define SAMPLE_SEP ";"
#define FEATURE_SEP " "
#define LABEL_SEP " "
#define SAMPLE_SEP_SIZE 2

using namespace sub_dl;

void load_mnist_data(const char*file_name,
    std::vector<std::vector<matrix_double> >& train_x_feature,
    std::vector<matrix_double>& train_y_label) {
    std::ifstream fis(file_name);
    int id, r, g, b, label;
    char comma;
    
    while (fis >> label) {
        matrix_double label_data(1, 1);
        int i = 0;
        label_data[0][0] = label;
        std::vector<matrix_double> x_feature;
        matrix_double value(28, 28);
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                int v;
                fis >> comma;
                fis >> v;
                value[i][j] = v / 256.0;
            }
        }
        x_feature.push_back(value);
        train_x_feature.push_back(x_feature);
        train_y_label.push_back(label_data);
    }
    std::cout << "Load Data" << std::endl;
}

void digital_recognition(int argc, char*argv[]) {
    if (argc < 2) {
        std::cout << "[Usage] ./digital_recognition train_data" << std::endl;
        return ;
    }
    std::vector<std::vector<matrix_double> > train_x_features;
    std::vector<matrix_double> train_y_labels;
    load_mnist_data(argv[1], train_x_features, train_y_labels);
    
    std::vector<Layer*> layers;
    layers.push_back(new ConvLayer(1, 6, 5, 5, 24, 24));
    layers.push_back(new ReluLayer());
    layers.push_back(new PoolingLayer(6, 6, 2, 2, 12, 12));
    layers.push_back(new ConvLayer(6, 6, 5, 5, 8, 8));
    layers.push_back(new ReluLayer());
    layers.push_back(new PoolingLayer(6, 6, 2, 2, 4, 4));
    layers.push_back(new ConvLayer(6, 10, 2, 2, 3, 3));
    layers.push_back(new ReluLayer());
    layers.push_back(new FlattenLayer());
    layers.push_back(new FullConnLayer(90, 32));
    layers.push_back(new SigmoidLayer());
    layers.push_back(new FullConnLayer(32, 10));
    layers.push_back(new SigmoidLayer());
    
    NetWrapper<MeanSquareLossLayer> *cnet = new NetWrapper<MeanSquareLossLayer>(10);
    cnet->_build_net(layers);
    int _max_epoch_cnt = 1000;
    int batch_size = 50;
    int tot = 10000;
    for (int epoch = 0; epoch < _max_epoch_cnt; epoch++) {
        for (int i = 0; i < tot / batch_size; i++) {
            double cost = 0.0;
            std::string val1, val2;
            std::vector<std::vector<matrix_double> > batch_x_features;
            std::vector<matrix_double> batch_y_labels;
            for (int j = i * batch_size; j < (i + 1) * batch_size 
                && j < train_y_labels.size(); j++) {
                batch_x_features.push_back(train_x_features[j]);
                batch_y_labels.push_back(train_y_labels[j]);
            }
            cost = cnet->_train(batch_x_features, batch_y_labels);
            // std::vector<int> labels;
            /*
            rnet->_rnn_predict(batch_x_features[0], labels);
            for (int i = 0; i < labels.size(); i++) {
                std::cout << labels[i] << " ";
            }
            std::cout << std::endl;
            for (int i = 0; i < batch_y_labels[0]._y_dim; i++) {
                std::cout << batch_y_labels[0][0][i] << " ";
            }
            std::cout << std::endl;
            */
            DEBUG_LOG("Cost %lf %s %s", cost, val1.c_str(), val2.c_str());
        }
    }    
}

int main(int argc, char* argv[]) {
    srand((unsigned)time(NULL));
    digital_recognition(argc, argv);
    return 0;
}
