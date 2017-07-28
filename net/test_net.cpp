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
void load_mnist_data(const char*file_name,
	std::vector<std::vector<matrix_double> >& train_x_feature,
	std::vector<matrix_double>& train_y_label) {
	std::ifstream fis(file_name);
	int id, r, g, b, label;
	char comma;
	
	while (fis >> label) {
		matrix_double label_data(1, 10);
		int i = 0;
		label_data[0][label] = 1;
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

void test_cnn(int argc, char*argv[]) {
    if (argc < 2) {
        std::cout << "[Usage] ./test_conv_layer train_data" << std::endl;
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
	
	NetworkWrapper<MeanSquareLossLayer> *cnet = new NetworkWrapper<MeanSquareLossLayer>(10);
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
            cost = cnet->_cnn_train(batch_x_features, batch_y_labels);
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


void test_rnn() {
    
    std::vector<Layer*> layers;
    layers.push_back(new WordEmbeddingLayer(14));
    //layers.push_back(new RnnCell(14, 8));
    //layers.push_back(new RnnCell(8, 16));
    
    // layers.push_back(new LstmCell(14, 8, true));
    // layers.push_back(new RnnCell(8, 16));
    
    // layers.push_back(new BiLstmCell(14, 16, false, true));    
    // layers.push_back(new BiLstmCell(16, 16, false, true));    
    //layers.push_back(new BiRnnCell<RnnCell>(14, 16, BI_RNN_CELL));
    // layers.push_back(new BiCellWrapper<LstmCell>(14, 32, false, false, BI_LSTM_CELL));
    // layers.push_back(new BiCellWrapper<RnnCell>(32, 32, BI_RNN_CELL));
    // layers.push_back(new GruCell(14, 16));
    // layers.push_back(new GruCell(16, 32));
    
    layers.push_back(new BiCellWrapper<RnnCell>(14, 16, BI_RNN_CELL));
    layers.push_back(new BiCellWrapper<GruCell>(16, 16, BI_GRU_CELL));
    layers.push_back(new BiCellWrapper<LstmCell>(16, 32, true, true, BI_LSTM_CELL));
    layers.push_back(new SeqFullConnLayer(32, 4));
    layers.push_back(new SeqActiveLayer());
    NetworkWrapper<SeqLossLayer> *rnet = new NetworkWrapper<SeqLossLayer>(4);
    rnet->_build_net(layers);
    // rnet->_load_feature_data();
    // load_data(rnet, "train_text.1");
    std::vector<matrix_double> train_x_features;
    std::vector<matrix_double> train_y_labels;
    load_data("train_text.seg.10w", train_x_features, train_y_labels);
    // load_feature_data(train_x_features, train_y_labels);
    // load_data("train_text.1", train_x_features, train_y_labels);
    int _max_epoch_cnt = 1000;
    int batch_size = 50;
    int tot = 100000;
    for (int epoch = 0; epoch < _max_epoch_cnt; epoch++) {
        for (int i = 0; i < tot / batch_size; i++) {
            double cost = 0.0;
            std::string val1, val2;
            std::vector<matrix_double> batch_x_features;
            std::vector<matrix_double> batch_y_labels;
            for (int j = i * batch_size; j < (i + 1) * batch_size 
                && j < train_y_labels.size(); j++) {
                batch_x_features.push_back(train_x_features[j]);
                batch_y_labels.push_back(train_y_labels[j]);
            }
            cost = rnet->_rnn_train(batch_x_features, batch_y_labels);
            std::vector<int> labels;
            rnet->_rnn_predict(batch_x_features[0], labels);
            for (int i = 0; i < labels.size(); i++) {
                std::cout << labels[i] << " ";
            }
            std::cout << std::endl;
            for (int i = 0; i < batch_y_labels[0]._y_dim; i++) {
                std::cout << batch_y_labels[0][0][i] << " ";
            }
            std::cout << std::endl;
            DEBUG_LOG("Cost %lf %s %s", cost, val1.c_str(), val2.c_str());
        }
    }    
}

int main(int argc, char* argv[]) {
    srand((unsigned)time(NULL));
    // test_rnn();
    test_cnn(argc, argv);
	return 0;
}
