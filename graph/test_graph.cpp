
#include "graph.h"
#include <stdarg.h> 
#include <string>
#include <fstream>
#include <vector>
#include <map>

using namespace sub_dl;

std::vector<int> REFER(int n, ...) {
    std::vector<int> ret;
	if (n == 0) {
		return ret;
	}

    va_list args;
    va_start(args, n);
    while (n > 0) {
        ret.push_back(va_arg(args, int));
        n --;
    }
    return ret;
}
#define SAMPLE_SEP ";"
#define FEATURE_SEP " "
#define LABEL_SEP " "
#define SAMPLE_SEP_SIZE 2

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

int main() {
    
    Graph *graph = new Graph();
	int input_1 = graph->_add_node(new DataFeedLayer(), REFER(0));
    int emb_id = graph->_add_node(new WordEmbeddingLayer(14), 
		REFER(1, input_1)); // input_2));
    int lstm_id = graph->_add_node(new LstmCell(14, 16, true), REFER(1, emb_id)); 
    int seq_full_id = graph->_add_node(new SeqFullConnSoftmaxLayer(16, 4), REFER(1, lstm_id));
	int loss_layer = graph->_add_node(new SeqCrossEntropyLossLayer(), REFER(1, seq_full_id));
    
	std::vector<std::vector<matrix_double> > train_x_features;
    std::vector<matrix_double> train_y_labels;
    load_data("train_text.seg.10w", train_x_features, train_y_labels);
    int _max_epoch_cnt = 1000;
    int batch_size = 50;
    int tot = 500;
	clock_t startTime,endTime;
    for (int epoch = 0; epoch < _max_epoch_cnt; epoch++) {
		startTime = clock();
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
        	graph->_run(batch_x_features, batch_y_labels, 4);
		}
		endTime = clock();
		std::cout << "Totle Time : " <<(double)(endTime - startTime) * 1000 / CLOCKS_PER_SEC << "ms" << std::endl;
    }

	return 0;
}
