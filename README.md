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



