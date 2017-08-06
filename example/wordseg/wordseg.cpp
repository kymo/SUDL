/* test_rnn.cpp */

#include <iostream>
#include <fstream>
#include "net_wrapper.h"

#define SAMPLE_SEP ";"
#define FEATURE_SEP " "
#define LABEL_SEP " "
#define SAMPLE_SEP_SIZE 2

using namespace sub_dl;

void load_data(const char*file_name,
    std::vector<std::vector<matrix_double> >& _train_x_features,
    std::vector<matrix_double>& _train_y_labels) {

    std::ifstream fis(file_name);
    std::string line;

    while (getline(fis, line)) {
        std::vector<std::string> split_strs;
        split(line, SAMPLE_SEP, split_strs);
        if (split_strs.size() != SAMPLE_SEP_SIZE) {
            std::cerr << "Error when load feature!" << std::endl;
            exit(1);
        }
        std::vector<std::string> features;
        std::vector<std::string> labels;
        split(split_strs[0], FEATURE_SEP, features);
        split(split_strs[1], LABEL_SEP, labels);
        if (features.size() != labels.size()) {
            std::cerr << "Error when load feature: size of features are \
                not match with size of labels!" << std::endl;
            exit(1);
        }
        std::vector<matrix_double> feature(features.size(), matrix_double(1, 1));
        matrix_double label(features.size(), 1);
        for (int i = 0; i < features.size(); i++) {
            feature[i][0][0] = atoi(features[i].c_str());
            // feature[0][i] = atoi(features[i].c_str());
            label[i][0] = atoi(labels[i].c_str());
        }
        _train_x_features.push_back(feature);
        _train_y_labels.push_back(label);
    }
    std::cout << "Load data done!" << std::endl;
}

void test_rnn() {
    
    std::vector<Layer*> layers;
    layers.push_back(new WordEmbeddingLayer(14));
    layers.push_back(new BiCellWrapper<RnnCell>(14, 16, BI_RNN_CELL));
    layers.push_back(new BiCellWrapper<GruCell>(16, 16, BI_GRU_CELL));
    layers.push_back(new BiCellWrapper<LstmCell>(16, 32, true, true, BI_LSTM_CELL));
    layers.push_back(new SeqFullConnLayer(32, 4));
    layers.push_back(new SeqActiveLayer());
    NetWrapper<SeqCrossEntropyLossLayer>*rnet = new NetWrapper<SeqCrossEntropyLossLayer>(4);
    rnet->_build_net(layers);
    std::vector<std::vector<matrix_double> > train_x_features;
    std::vector<matrix_double> train_y_labels;
    load_data("train_text.seg.10w", train_x_features, train_y_labels);
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
            cost = rnet->_train(batch_x_features, batch_y_labels);
            std::vector<int> labels;
            rnet->_predict(batch_x_features[0], labels);
            for (int i = 0; i < labels.size(); i++) {
                std::cout << labels[i] << " ";
            }
            std::cout << std::endl;
            for (int i = 0; i < batch_y_labels[0]._x_dim; i++) {
                std::cout << batch_y_labels[0][i][0] << " ";
            }
            std::cout << std::endl;
            DEBUG_LOG("Cost %lf %s %s", cost, val1.c_str(), val2.c_str());
        }
    }    
}

int main() {
    srand((unsigned)time(NULL));
    test_rnn();
    return 0;
}
