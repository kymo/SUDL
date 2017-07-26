
#include <iostream>
#include "seq_full_conn_layer.h"
#include "seq_loss_layer.h"
#include "rnn_cell.h"
#include "matrix.h"
#include "util.h"
#include <fstream>

#define EMBEDDING_DIM 14
#define LABEL_DIM 4
#define SAMPLE_SEP ";"
#define FEATURE_SEP " "
#define LABEL_SEP " "
#define SAMPLE_SEP_SIZE 2

using namespace sub_dl;

class ReccurentNet {

public:

	int _input_dim;
	int _output_dim;

	ReccurentNet() {
		// _input_dim = input_dim;
		// _output_dim = output_dim;
	}

	std::vector<Layer*> _layers;
	std::vector<matrix_double> _train_x_features;
	std::vector<matrix_double> _train_y_labels;


	void _build_rnn(const std::vector<Layer*> layers) {
		_layers = layers;
	}

	void _forward(const matrix_double& feature) {
		std::vector<matrix_double> x;
		for (int i = 0; i < feature._x_dim; i++) {
			x.push_back(feature._R(i));
		}
		DataFeedLayer* data_layer = new DataFeedLayer(x);
		Layer* pre_layer = data_layer;
		for (auto layer : _layers) {
			layer->_forward(pre_layer);
			pre_layer = layer;
		}
	}
    
	void _push_feature(const matrix_double& feature,
        const matrix_double& label) {
        _train_x_features.push_back(feature);
        _train_y_labels.push_back(label);
    }

	void _update_gradient() {
		for (auto layer : _layers) {
			layer->_update_gradient(SGD, -0.01);
		}
	}

	void gradient_check(const matrix_double& feature,
		const matrix_double& label
		) {
		
		for (int l = 0; l < _layers.size(); l++) {
			if (_layers[l]->_type == SEQ_FULL) {
				std::cout << "------------Gradient Check for seq full layer -------------" << std::endl;
				SeqFullConnLayer* seq_full = (SeqFullConnLayer*) _layers[l];
				for (int i = 0; i < seq_full->_input_dim; i++) {
					for (int j = 0; j < seq_full->_output_dim; j++) {
						double v = seq_full->_seq_full_weights[i][j];
						seq_full->_seq_full_weights[i][j] = v + 1.0e-4;
						_forward(feature);
						Layer* nxt_layer = new SeqLossLayer(label);
						nxt_layer->_forward(_layers.back());
						nxt_layer->_backward(NULL);
						double f1 = 0.0;
						for (int k = 0; k < nxt_layer->_seq_len; k++) {
							f1 += nxt_layer->_data[k].sum();
						}

						seq_full->_seq_full_weights[i][j] = v - 1.0e-4;
						_forward(feature);
						nxt_layer = new SeqLossLayer(label);
						nxt_layer->_forward(_layers.back());
						nxt_layer->_backward(NULL);
						double f2 = 0.0;
						for (int k = 0; k < nxt_layer->_seq_len; k++) {
							f2 += nxt_layer->_data[k].sum();
						}
						std::cout << "[ " << seq_full->_delta_seq_full_weights[i][j] << "," << (f1 - f2) / (2.0e-4) << "]";
						seq_full->_seq_full_weights[i][j] = v;
					}
					std::cout << std::endl;
				}
				int i = 0;
				std::cout << "bias" << std::endl;	
				for (int j = 0; j < seq_full->_output_dim; j++) {
					
					double v = seq_full->_seq_full_bias[i][j];
					seq_full->_seq_full_bias[i][j] = v + 1.0e-4;
					_forward(feature);
					Layer* nxt_layer = new SeqLossLayer(label);
					nxt_layer->_forward(_layers.back());
					nxt_layer->_backward(NULL);
					double f1 = 0.0;
					for (int k = 0; k < nxt_layer->_seq_len; k++) {
						f1 += nxt_layer->_data[k].sum();
					}

					seq_full->_seq_full_bias[i][j] = v - 1.0e-4;
					_forward(feature);
					nxt_layer = new SeqLossLayer(label);
					nxt_layer->_forward(_layers.back());
					nxt_layer->_backward(NULL);
					double f2 = 0.0;
					for (int k = 0; k < nxt_layer->_seq_len; k++) {
						f2 += nxt_layer->_data[k].sum();
					}
					std::cout << "[ " << seq_full->_delta_seq_full_bias[i][j] << "," << (f1 - f2) / (2.0e-4) << "]";
					seq_full->_seq_full_bias[i][j] = v;					
				}
				std::cout << std::endl;
			} else if (_layers[l]->_type == RNN_CELL) {
				std::cout << "------------Gradient Check for rnn cell layer -------------" << std::endl;
				
				RnnCell* rnn_cell = (RnnCell*) _layers[l];
				for (int i = 0; i < rnn_cell->_input_dim; i++) {
					for (int j = 0; j < rnn_cell->_output_dim; j++) {
						double v = rnn_cell->_input_hidden_weights[i][j];
						rnn_cell->_input_hidden_weights[i][j] = v + 1.0e-4;
						_forward(feature);
						Layer* nxt_layer = new SeqLossLayer(label);
						nxt_layer->_forward(_layers.back());
						nxt_layer->_backward(NULL);
						double f1 = 0.0;
						for (int k = 0; k < nxt_layer->_seq_len; k++) {
							f1 += nxt_layer->_data[k].sum();
						}

						rnn_cell->_input_hidden_weights[i][j] = v - 1.0e-4;
						_forward(feature);
						nxt_layer = new SeqLossLayer(label);
						nxt_layer->_forward(_layers.back());
						nxt_layer->_backward(NULL);
						double f2 = 0.0;
						for (int k = 0; k < nxt_layer->_seq_len; k++) {
							f2 += nxt_layer->_data[k].sum();
						}
						std::cout << "[ " << rnn_cell->_delta_input_hidden_weights[i][j] << "," << (f1 - f2) / (2.0e-4) << "]";
						rnn_cell->_input_hidden_weights[i][j] = v;
					}
					std::cout << std::endl;
				}
			}
		}
	}
	double _backward(const matrix_double& label) {

		SeqLossLayer* loss_layer = new SeqLossLayer(label);
		loss_layer->_forward(_layers.back());
		loss_layer->_backward(NULL);
		Layer* nxt_layer = loss_layer;
		double cost = 0.0;
		for (int i = 0; i < loss_layer->_seq_len; i++) {
			cost += loss_layer->_data[i].sum();
		}
		for (int j = _layers.size() - 1; j >= 0; j--) {
			// nxt_layer->display();
			_layers[j]->_backward(nxt_layer);
			nxt_layer = _layers[j];
		}
		// nxt_layer->display();
		return cost;
	}

	void _train() {
		int _max_epoch_cnt = 1000;
		int batch_size = 50;
		int tot = 10000;
		for (size_t epoch = 0; epoch < _max_epoch_cnt; epoch++) {
			for (int i = 0; i < tot / batch_size; i++) {
				std::vector<int> sample_indexes;
				double cost = 0.0;
				std::string val1, val2;
				for (int j = i * batch_size; j < (i + 1) * batch_size; j++) {
					const matrix_double& feature = _train_x_features[j];
					const matrix_double& label = _train_y_labels[j];
					_forward(feature);
					// val1 = merve(label, 1);
					val1 = merge(label, 1);
					matrix_double output_val(feature._x_dim,
						label._y_dim);
					for (int i = 0; i < _layers.back()->_data.size(); i++) {
						output_val.set_row(i, _layers.back()->_data[i]);
					}
					val2 = merge(output_val, 1);
					// cost += _layers.back()->_data[0].sum() * 0.5;	
					cost += _backward(label);
					// gradient_check(feature, label);
					_update_gradient();
				}
				std::cout << "Cost " << cost << " " << val1 << " " << val2 << std::endl;
			}
		}
	
	}

	void _load_feature_data() {
		// load data
		for (size_t i = 0; i < 12500; i++) {    
			matrix_double x(2, _input_dim);
			matrix_double y(2, _output_dim);
			int sum = 0;
			for (size_t j = 0; j < _input_dim; j ++) {
				int v = rand() % 2;
				//int v;
				//fis >> v;
				sum += v;
				size_t k = 0;
				while (v > 0) {
					x[k][j] = v % 2;
					v /= 2;
					k += 1;
				}
				for (; k < 2;k ++) {
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
			for (; k < 2;k ++) {
				y[k][0] = 0;
			}
			if (i < 10000) {
				_train_x_features.push_back(x);
				_train_y_labels.push_back(y);
			}
		}
		std::cout << "laod data okay!" << std::endl;
	}

};

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
	std::cout << file_name << std::endl;
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
	
	std::vector<Layer*> layers;
	layers.push_back(new RnnCell(EMBEDDING_DIM, 8));
	layers.push_back(new RnnCell(8, 16));
	layers.push_back(new SeqFullConnLayer(16, LABEL_DIM));
	layers.push_back(new SeqActiveLayer());
	
	ReccurentNet *rnet = new ReccurentNet();
	rnet->_build_rnn(layers);
	// rnet->_load_feature_data();
	// load_data(rnet, "train_text.1");
	load_data(rnet, "train_text.seg.10w");
	rnet->_train();

}

int main() {
	srand((unsigned)time(NULL));
	test_rnn();
	return 0;
}
