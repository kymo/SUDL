/* test_rnn.cpp */

#include <iostream>
#include <fstream>
#include "util.h"
#include "reccurent_net.h"

#define SAMPLE_SEP ";"
#define FEATURE_SEP " "
#define LABEL_SEP " "
#define SAMPLE_SEP_SIZE 2

using namespace sub_dl;

void load_data(const char*file_name,
    std::vector<matrix_double>& _train_x_features,
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
        matrix_double feature(1, features.size());
        matrix_double label(1, features.size());
        for (int i = 0; i < features.size(); i++) {
            feature[0][i] = atoi(features[i].c_str());
            label[0][i] = atoi(labels[i].c_str());
        }
        _train_x_features.push_back(feature);
        _train_y_labels.push_back(label);
    }
    std::cout << "Load data done!" << std::endl;
}
void load_feature_data(std::vector<matrix_double>& _train_x_features,
    std::vector<matrix_double>& _train_y_labels) {
    // load data
    int _feature_dim = 2;
    int _output_dim = 1;
    for (size_t i = 0; i < 12500; i++) {    
        matrix_double x(8, _feature_dim);
        matrix_double y(8, _output_dim);
        int sum = 0;
        for (size_t j = 0; j < _feature_dim; j ++) {
            int v = rand() % 128;
            //int v;
            //fis >> v;
            sum += v;
            size_t k = 0;
            while (v > 0) {
                x[k][j] = v % 2;
                v /= 2;
                k += 1;
            }
            for (; k < 8;k ++) {
                x[k][j] = 0;
            }
        }
        //fis >> sum;
        size_t k = 0;
        while (sum > 0) {
            y[k][0] = sum % 2;
            sum /= 2;
            k += 1;
        }
        for (; k < 8;k ++) {
            y[k][0] = 0;
        }
        if (i < 10000) {
            _train_x_features.push_back(x);
            _train_y_labels.push_back(y);
        }
    }
}


void test_rnn() {
    
    std::vector<Layer*> layers;
    layers.push_back(new WordEmbeddingLayer(14));
    //layers.push_back(new RnnCell(14, 8));
    //layers.push_back(new RnnCell(8, 16));
	layers.push_back(new LstmCell(14, 8, true));
	layers.push_back(new RnnCell(8, 16));
    layers.push_back(new SeqFullConnLayer(16, 4));
    layers.push_back(new SeqActiveLayer());
    ReccurentNet *rnet = new ReccurentNet(4);
    rnet->_build_rnn(layers);
    // rnet->_load_feature_data();
    // load_data(rnet, "train_text.1");
    std::vector<matrix_double> train_x_features;
    std::vector<matrix_double> train_y_labels;
    // load_data("train_text.seg.10w", train_x_features, train_y_labels);
    // load_feature_data(train_x_features, train_y_labels);
    load_data("train_text.1", train_x_features, train_y_labels);
    int _max_epoch_cnt = 100;
    int batch_size = 10;
    int tot = 10000;
    for (int epoch = 0; epoch < _max_epoch_cnt; epoch++) {
        for (int i = 0; i < tot / batch_size; i++) {
            double cost = 0.0;
            std::string val1, val2;
            std::vector<matrix_double> batch_x_features;
            std::vector<matrix_double> batch_y_labels;
            for (int j = i * batch_size; j < (i + 1) * batch_size; j++) {
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
			for (int i = 0; i < batch_y_labels[0]._y_dim; i++) {
				std::cout << batch_y_labels[0][0][i] << " ";
			}
			std::cout << std::endl;
			std::cout << "Cost " << cost << " " << val1 << " " << val2 << std::endl;
        }
    }    
}

int main() {
    srand((unsigned)time(NULL));
    test_rnn();
    return 0;
}
