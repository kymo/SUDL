#ifndef LAYER_FACTORY_H
#define LAYER_FACTORY_H

#include <iostream>
#include "sudl.pb.h"
#include "layer.h"
#include "loss_layer.h"
#include "seq_full_conn_layer.h"
#include "seq_full_conn_softmax_layer.h"
#include "seq_loss_layer.h"
#include "rnn_cell.h"
#include "lstm_cell.h"
#include "gru_cell.h"
#include "bi_cell_wrapper.h"
#include "matrix.h"
#include "conv_layer.h"
#include "pooling_layer.h"
#include "full_conn_layer.h"
#include "full_conn_softmax_layer.h"
#include "active_layer.h"
#include "loss_layer.h"

namespace sub_dl {

class LayerFactory {
    
private:
    static LayerFactory* _instance;
    LayerFactory() {}

public:

    static LayerFactory* _get_instance() {
        if (NULL == _instance) {
            _instance = new LayerFactory();
        }
        return _instance;
    }
    
    // static std::shared_ptr<Layer>
    static Layer* _produce(const lm::LayerParam& layer_param) {
        std::string layer_type = layer_param.type();
        Layer* layer;
        if (layer_type == "DataFeedLayer") {
			layer = new DataFeedLayer();
		} else if (layer_type == "WordEmbeddingLayer") {
			layer = new WordEmbeddingLayer(layer_param.fc_param().output_dim());
		} else if (layer_type == "LstmCell") {
			layer = new LstmCell(layer_param.rnn_cell_param());
			// layer_param.input_dim(),
			//	layer_param.output_dim(), layer_param.use_peephole());
		} else if (layer_type == "SeqFullConnSoftmaxLayer") {
			layer = new SeqFullConnSoftmaxLayer(layer_param.fc_param());
			//layer_param.input_dim(),
			//	layer_param.output_dim());
		} else if (layer_type == "SeqCrossEntropyLossLayer") {
            layer = new SeqCrossEntropyLossLayer();
        } else if (layer_type == "ConvLayer") {
			layer = new ConvLayer(layer_param.conv_param());
		} else if (layer_type == "PoolingLayer") {
			layer = new PoolingLayer(layer_param.pool_param());
		} else if (layer_type == "ReluLayer") {
			layer = new ReluLayer();
		} else if (layer_type == "FlattenLayer") {
			layer = new FlattenLayer();
		} else if (layer_type == "FullConnLayer") {
			layer = new FullConnLayer(layer_param.fc_param());
		} else if (layer_type == "SigmoidLayer") {
			layer = new SigmoidLayer();
		} else if (layer_type == "FullConnSoftmaxLayer") {
			layer = new FullConnSoftmaxLayer(layer_param.fc_param());
		} else if (layer_type == "CrossEntropyLossLayer") {
			layer = new CrossEntropyLossLayer();
		}
        return layer;
        // return std::shared_ptr<Layer>(layer);
    }

};

LayerFactory* LayerFactory::_instance = NULL;
#define CREATER_LAYER(layer_param) \
    LayerFactory::_get_instance()->_produce(layer_param)

}

#endif
