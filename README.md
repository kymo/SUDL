# SUDL

A light deep learning tools box by c++

## Contains

**Network Architecture**
1. Convolutional Neural Network 
2. Normal Neural Network
3. Reccurent Neural Network with three mainstream varieties(LSTM, LSTM-peelhole, GRU)(deep architecture supported)
4. bi-directional LSTM(peephole) & GRU & RNN (deep architecture supported)

**Nonlinearities**
1. ReLU
2. Sigmoid
3. tanh

**TODO**
1. GPU supported (No Gpu for testing :( )
2. network architecture configurable by proto Done (protoc is needed to be installed first)

**Compile**

sh build.sh(cmake is needed)

**Usage**

net architecture is built by proto file that you defined, just like what the examples do.

rnn.prototxt

    name: "test"
    layer {
        name: "DataFeedLayer"
        type: "DataFeedLayer"
        top: "input_data"
    }

    layer {
        name: "WordEmbeddingLayer"
        type: "WordEmbeddingLayer"
        top: "emb1"
        bottoms: "input_data"
        fc_param {
            output_dim: 14
            input_dim: 0
        }
    }

    layer {
        name: "LstmCell"
        type: "LstmCell"
        top: "lstm1"
        bottoms: "emb1"

        rnn_cell_param {
            input_dim: 14
            output_dim: 16
            use_peephole: false
        }
    }

    layer {
        name: "LstmCell1"
        type: "LstmCell"
        top: "lstm2"
        bottoms: "lstm1"
        rnn_cell_param {
            input_dim: 16
            output_dim: 16
            use_peephole: true
        }
    }

    layer {
        name: "SeqFullConnSoftmaxLayer"
        type: "SeqFullConnSoftmaxLayer"
        top: "seqsoftmax1"
        bottoms: "lstm2"
        fc_param {
            input_dim: 16
            output_dim: 4
        }

    }
    layer {
        name: "SeqCrossEntropyLossLayer"
        type: "SeqCrossEntropyLossLayer"
        top: "loss"
        bottoms: "seqsoftmax1"
    }


cnn.prototxt

    name: "cnn"
    
    layer {
        name: "DataFeedLayer"
        type: "DataFeedLayer"
        top: "input_data"
    }

    layer {
        name: "ConvLayer1"
        type: "ConvLayer"
        bottoms: "input_data"
        top: "conv1"

        conv_param {
            input_dim: 1
            output_dim: 2
            kernel_x_dim: 11
            kernel_y_dim: 11
            feature_x_dim: 18
            feature_y_dim: 18
        }

    }

    layer {
        name: "ReluLayer1"
        type: "ReluLayer"
        bottoms: "conv1"
        top: "relu1"
    }

    layer {
        name: "PoolingLayer1"
        type: "PoolingLayer"
        bottoms: "relu1"
        top: "pool1"

        pool_param {
            input_dim: 2
            output_dim: 2
            pooling_x_dim: 2
            pooling_y_dim: 2
            feature_x_dim: 9
            feature_y_dim: 9
        }

    }

    layer {
        name: "ConvLayer2"
        type: "ConvLayer"
        bottoms: "pool1"
        top: "conv2"

        conv_param {
            input_dim: 2
            output_dim: 2
            kernel_x_dim: 4
            kernel_y_dim: 4
            feature_x_dim: 6
            feature_y_dim: 6
        }
    }

    layer {
        name: "ReluLayer2"
        type: "ReluLayer"
        bottoms: "conv2"
        top: "relu2"
    }


    layer {
        name: "FlattenLayer1"
        type: "FlattenLayer"
        bottoms: "relu2"
        top: "flat1"
    }

    layer {
        name: "FullConnLayer"
        type: "FullConnLayer"
        bottoms: "flat1"
        top: "full1"

        fc_param {
            input_dim: 72
            output_dim: 32
        }
    }

    layer {
        name: "SigmoidLayer"
        type: "SigmoidLayer"
        bottoms: "full1"
        top: "sigmoid1"

    }


    layer {
        name: "FullConnSoftmaxLayer"
        type: "FullConnSoftmaxLayer"
        bottoms: "sigmoid1"
        top: "full2"

        fc_param {
            input_dim: 32
            output_dim: 10
        }
    }

    layer {
        name: "loss"
        type: "CrossEntropyLossLayer"
        bottoms: "full2"
        top: "cross"
    }
