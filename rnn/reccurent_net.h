#ifndef RECCURENT_NET_H
#define RECCURENT_NET_H

#include <iostream>
#include "seq_full_conn_layer.h"
#include "seq_loss_layer.h"
#include "rnn_cell.h"
#include "matrix.h"
#include "embedding_layer.h"
#include "util.h"
#include <fstream>

#define EMBEDDING_DIM 14
#define LABEL_DIM 4
#define SAMPLE_SEP ";"
#define FEATURE_SEP " "
#define LABEL_SEP " "
#define SAMPLE_SEP_SIZE 2


namespace sub_dl {

class ReccurentNet {

public:
    // input dim
    int _input_dim;
    // output dim
    int _output_dim;

    ReccurentNet(int output_dim) {
        _output_dim = output_dim;
    }

    // layers
    std::vector<Layer*> _layers;

    // build reccurent net
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
            _layers[j]->_backward(nxt_layer);
            nxt_layer = _layers[j];
        }
        return cost;
    }

    double _train(const std::vector<matrix_double>& batch_x,
        const std::vector<matrix_double>& batch_y) {
        
        double cost = 0.0;
        std::string val1, val2;
        for (int i = 0; i < batch_x.size(); i++) {
            const matrix_double& feature = batch_x[i];
            const matrix_double& label_ids = batch_y[i];
            matrix_double label;
            label_encode(label_ids, label, _output_dim);
            // forward
            _forward(feature);
            val1 = merge(label, 1);
            matrix_double output_val(feature._x_dim,
                label._y_dim);
            for (int i = 0; i < _layers.back()->_data.size(); i++) {
                output_val.set_row(i, _layers.back()->_data[i]);
            }
            val2 = merge(output_val, 1);
            cost += _backward(label);
            // gradient_check(feature, label);
            _update_gradient();
        }
        return cost;
    }

	void _predict(const matrix_double& feature, std::vector<int>& labels) {
		_forward(feature);
		const Layer* layer = _layers.back();
		std::cout << layer->_data.size() << std::endl;
		std::cout << layer->_type << std::endl;
		for (int i = 0; i < layer->_data.size(); i++) {
			int mx_id = 0;
			float val = -100;
			for (int j = 0; j < layer->_data[i]._y_dim; j++) {
				if (val < layer->_data[i][0][j]) {
					val = layer->_data[i][0][j];
					mx_id = j;
				}
			}
			std::cout << mx_id << " ";
			labels.push_back(mx_id);
		}
		std::cout << std::endl;
	}

};

}

#endif
