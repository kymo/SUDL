#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include "rnn.h"
#include "lstm.h"
#include "gru.h"
#define EMBEDDING_DIM 14
#define LABEL_DIM 4
#define SAMPLE_SEP ";"
#define FEATURE_SEP " "
#define LABEL_SEP " "
#define SAMPLE_SEP_SIZE 2

using namespace sub_dl;

void split(const std::string& str, 
		const std::string& delim,
		std::vector<std::string>& ret) {
	if (str.size() <= 0 || delim.size() <= 0) {
		ret.clear();
		return;
	}
	ret.clear();
	int last = 0;
	int index = str.find_first_of(delim, last);
	while (index != -1) {
		ret.push_back(str.substr(last, index - last));
		last = index + 1;
		index = str.find_first_of(delim, last);
	}
	if (index == -1 && str[str.size() - 1] != '\t') {
		ret.push_back(str.substr(last, str.size() - last));
	}
}

void word_embedding(std::vector<int> words_id_vec,
		matrix_double& feature) {
	feature.resize(words_id_vec.size(), EMBEDDING_DIM); 
	for (int i = 0; i < words_id_vec.size(); i++) {
		int v = words_id_vec[i];
		int t = 0;
		while (v) {
			feature[i][t ++] = v % 2;
			v /= 2;
		}
		for (; t < EMBEDDING_DIM; t++) {
			feature[i][t] = 0;
		}
	}

}

void label_encode(std::vector<int> label_id_vec,
		matrix_double& label) {
	label.resize(label_id_vec.size(), LABEL_DIM);
	for (int i = 0; i < label_id_vec.size(); i++) {
		label[i][label_id_vec[i]] = 1;
	}
}


template <typename T>
void load_data(T* rnn, const char*file_name) {

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
		std::vector<int> word_id_vec;
		std::vector<int> label_id_vec;
		for (int i = 0; i < features.size(); i++) {
			word_id_vec.push_back(atoi(features[i].c_str()));
			label_id_vec.push_back(atoi(labels[i].c_str()));
		}
		matrix_double feature;
		word_embedding(word_id_vec, feature);
		matrix_double label;
		label_encode(label_id_vec, label);
		rnn->_push_feature(feature, label);
	}
	std::cout << "Load data done!" << std::endl;
}

void test_rnn() {
	RNN* rnn = new RNN(EMBEDDING_DIM, 128, LABEL_DIM);
	rnn->_set_epoch_cnt(100);
	rnn->_set_eta(-0.005);
	rnn->_set_clip_gra(0.1);
	load_data(rnn, "train_text.seg.50w");
	rnn->_train();

}

void test_lstm() {
	LSTM* rnn = new LSTM(EMBEDDING_DIM, 128, LABEL_DIM, true);
	rnn->_set_epoch_cnt(100);
	rnn->_set_eta(-0.02);
	rnn->_set_clip_gra(0.2);
	load_data(rnn, "train_text.1");
	rnn->_train();
}

void test_gru() {
	GRU* rnn = new GRU(EMBEDDING_DIM, 128, LABEL_DIM);
	rnn->_set_epoch_cnt(100);
	rnn->_set_eta(-0.02);
	rnn->_set_clip_gra(0.2);
	load_data(rnn, "train_text.1");
	rnn->_train();
}

void test_gru_add() {
    GRU *gru = new GRU(2, 16, 1);
    gru->_set_epoch_cnt(100);
    gru->_set_eta(-0.05);
	gru->_set_clip_gra(5);
    gru->_load_feature_data();
    gru->_train();
}

int main () {
	//test_lstm();
	test_rnn();
	//test_gru();
	return 0;
}
