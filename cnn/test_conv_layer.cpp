#include "layer.h"
#include "conv_layer.h"
#include "pooling_layer.h"
#include <time.h>
#include <iostream>
using namespace sub_dl;

class InputLayer: public Layer {

public:
InputLayer() {}
    void _forward(Layer* pre_layer) {}
    void _backward(Layer* nxt_laery) {}
    void _add_data(const matrix_double& data) {
        _data.push_back(data);
    }
};

void test_conv_layer() {
    
    matrix_double data(3, 3);
    int a[3][3] = {{1,2,3}, {3,0,1}, {-1,2,1}};
    for(int i = 0; i < 3;i ++) {
        for (int j = 0; j < 3;j ++) {
            data[i][j] = a[i][j];
        }
    }
    matrix_int conn_map(1, 2);
    for (int i = 0; i < 2; i++) {
        conn_map[0][i] = 1;
    }
    InputLayer* layer = new InputLayer();
    layer->_add_data(data);
    ConvLayer* conv_layer = new ConvLayer(1, 2, 2, 2, 2, 2);
    conv_layer->_set_conn_map(conn_map);
    conv_layer->_forward(layer);
    conv_layer->display();
    
    PoolingLayer* pooling_layer = new PoolingLayer(2, 2, 2, 2, 1, 1);
    pooling_layer->_forward(conv_layer);
    pooling_layer->display();
  
    matrix_double error(1,1);
    error[0][0] = 0.32;
    pooling_layer->_errors.push_back(error);
    error[0][0] = -0.21;
    pooling_layer->_errors.push_back(error);
    // test backward of conv layer
    conv_layer->_backward(pooling_layer);
}

void test_pooling_layer() {

	InputLayer* layer = new InputLayer();
	for (int i = 0; i < 4; i ++) {
		matrix_double data(6, 6);
		for (int j = 0; j < 6; j++) {
			for (int k = 0; k < 6; k++) {
				data[j][k] = 2 * (rand() % 4) - 4;
			}
		}
		layer->_add_data(data);
		data._display("data");
	}

	PoolingLayer* pooling_layer = new PoolingLayer(4, 4, 2, 2, 3, 3);
	pooling_layer->_forward(layer);
	pooling_layer->display();
	
	ConvLayer* conv_layer = new ConvLayer(4, 2, 2, 2, 2, 2);
	
	int link[4][2] = {{1, 0}, {1, 0}, {0, 1}, {0, 1}};
	matrix_int conn_map(4, 2);
	for (int i = 0; i < 4; i++){
		for (int j = 0; j < 2; j++) {
			conn_map[i][j] = link[i][j];
		}
	}
	conv_layer->_set_conn_map(conn_map);
	conv_layer->_forward(pooling_layer);
	conv_layer->display();

	matrix_double error1(2, 2);
	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 2; j++) {
			error1[i][j] = (rand() % 10 + 1)/10.0;
		}
	}
	error1._display("error1");
	conv_layer->_errors.push_back(error1);
	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 2; j++) {
			error1[i][j] = (rand() % 10 + 1)/10.0;
		}
	}
	error1._display("error2");
	conv_layer->_errors.push_back(error1);
	pooling_layer->_backward(conv_layer);
}


int main() {
    srand((unsigned)time(NULL));
    // test_conv_layer();    
    test_pooling_layer();
	return 0;
}
