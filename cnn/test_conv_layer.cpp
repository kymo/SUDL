#include "layer.h"
#include "conv_layer.h"
#include "pooling_layer.h"
#include "full_conn_layer.h"
#include "loss_layer.h"
#include "cnn.h"
#include "ann.h"
#include "active_func.h"
#include "active_layer.h"
#include "flat_layer.h"

#include <time.h>
#include <iostream>
using namespace sub_dl;
using namespace std;

class InputLayer: public Layer {

public:
    InputLayer(const std::vector<matrix_double>&datas) {
        _data = datas;    
    }
    void _forward(Layer* pre_layer) {}
    void _backward(Layer* nxt_laery) {}
    void _update_gradient(int opt_type, double learning_rate) {}
    void display() {

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
    std::vector<matrix_double> datas;
    datas.push_back(data);
    Layer* layer = new InputLayer(datas);
    
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
    // test backward of conv layer
    for (int i = 0; i < 2; i++) {
        conv_layer->_errors[i]._display("error");
    }
}

void test_conv_full_layer() {    
    /*
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
    std::vector<matrix_double> datas;
    datas.push_back(data);
    Layer* layer = new InputLayer(datas);
    
    ConvLayer* conv_layer = new ConvLayer(1, 2, 2, 2, 2, 2);
    conv_layer->_set_conn_map(conn_map);
    conv_layer->_forward(layer);
    conv_layer->display();
    
    FullConnLayer* full_layer = new FullConnLayer(8, 4);
    full_layer->_forward(conv_layer);
    full_layer->display();

    matrix_double error(1, 4);
    error[0][0] = 0.32;
    error[0][1] = 0.21;
    error[0][2] = 0.1;
    error[0][3] = 0.5;

    full_layer->_errors.push_back(error);
    // test backward of conv layer
    conv_layer->_backward(full_layer);
    
    for (int i = 0; i < 2;i ++) {
        conv_layer->_errors[i]._display("errors conv");
    }
    for (int i = 0; i < 2;i ++) {
        conv_layer->_delta_conv_kernels[0][i]._display("delta_weights");
    }
    full_layer->_full_conn_weights._display("_full_conn_weights");
    full_layer->_errors[0]._display("error");
	*/
}


void test_pooling_layer() {
    std::vector<matrix_double> datas;
    for (int i = 0; i < 4; i ++) {
        matrix_double data(6, 6);
        for (int j = 0; j < 6; j++) {
            for (int k = 0; k < 6; k++) {
                data[j][k] = 2 * (rand() % 4) - 4;
            }
        }
        //layer->_add_data(data);
        data._display("data");
        datas.push_back(data);
    }
    Layer* layer = new InputLayer(datas);

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

void test_cnn(int argc, char*argv[]) {
    if (argc < 2) {
        std::cout << "[Usage] ./test_conv_layer train_data" << std::endl;
        return ;
    }
	std::vector<Layer*> layers;
    layers.push_back(new ConvLayer(1, 6, 5, 5, 24, 24));
	layers.push_back(new ReluLayer());
    layers.push_back(new PoolingLayer(6, 6, 2, 2, 12, 12));
	layers.push_back(new SigmoidLayer());
    layers.push_back(new ConvLayer(6, 6, 5, 5, 8, 8));
	layers.push_back(new ReluLayer());
	layers.push_back(new FlatternLayer());
    //layers.push_back(new PoolingLayer(16, 16, 2, 2, 4, 4));
    //layers.push_back(new ConvLayer(16, 120, 4, 4, 1, 1));
    layers.push_back(new FullConnLayer(384, 32));
	layers.push_back(new SigmoidLayer());
    layers.push_back(new FullConnLayer(32, 10));
	layers.push_back(new SigmoidLayer());
    //layers.push_back(new ConvLayer(1, 1, 1, 1, 2, 2));
    //layers.push_back(new PoolingLayer(1, 1, 2, 2, 1, 1));
    //layers.push_back(new ConvLayer(1, 2, 1, 1, 1, 1));
    //layers.push_back(new FullConnLayer(2, 3));
    
    //layers.push_back(new PoolingLayer(1, 1, 4, 4, 7, 7));
    //layers.push_back(new ConvLayer(1, 1, 2, 2, 6, 6));
    //layers.push_back(new FullConnLayer(36, 10));
    //Layer* loss_layer = new MeanSquareLossLayer();
    //layers.push_back(loss_layer);
    CNN<MeanSquareLossLayer> *cnn = new CNN<MeanSquareLossLayer>();
    cnn->build_cnn(layers);
    cnn->load_data(argv[1]);
    //cnn->load_test_data(argv[1]);
    //cnn->load_iris_data(argv[1]);
    cnn->train();
}


void test_cnn_1() {
    int label = 2;
    matrix_double feature(7, 7);
    for (int i = 0; i < 7; i++) {
        for (int j = 0; j < 7; j++) {
            feature[i][j] = rand() % 10 / 10.0;
        }
    }
    std::vector<matrix_double> data;
    matrix_double la(1, 4);
    la[0][label] = 0;
    CNN<MeanSquareLossLayer> *cnn = new CNN<MeanSquareLossLayer>();
    data.push_back(feature);
    std::vector<Layer*> layers;
    layers.push_back(new ConvLayer(1, 2, 2, 2, 6, 6));
    //layers.push_back(new PoolingLayer(2, 2, 2, 2, 3, 3));
    //layers.push_back(new ConvLayer(2, 2, 2, 2, 2, 2));
    //layers.push_back(new FullConnLayer(8, 4));
    cnn->build_cnn(layers);
    cnn->train_x_feature.push_back(data);
    cnn->train_y_label.push_back(la);
    cnn->train();

}

void test_ann(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "[Usage] ./test_conv_layer train_data" << std::endl;
        return ;
    }
    std::vector<Layer*> layers;
    //layers.push_back(new FullConnLayer(49, 64));
    //layers.push_back(new FullConnLayer(64, 10));
    layers.push_back(new FullConnLayer(4, 8));
    layers.push_back(new ReluLayer());
	layers.push_back(new FullConnLayer(8, 32));
    layers.push_back(new SigmoidLayer());
    layers.push_back(new FullConnLayer(32, 3)); 
    layers.push_back(new SigmoidLayer());
	
	ANN<MeanSquareLossLayer> *ann = new ANN<MeanSquareLossLayer>();
    ann->build_ann(layers);
    ann->load_data(argv[1]);
    //ann->load_mnist_data(argv[1]);
    ann->train();
}

int main(int argc, char*argv[]) {
    srand((unsigned)time(NULL));
    //test_conv_layer();   
    // test_conv_full_layer();
    // test_pooling_layer();
    test_cnn(argc, argv);
    //test_ann(argc, argv);
    //test_cnn_1();
    return 0;
}
