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
1. GPU supported
2. network architecture configurable 

**Usage**
###compile:
sh build.sh

###different types of net:

1. ANN
> layers.push_back(new FullConnLayer(4, 8));  </br>
layers.push_back(new ReluLayer());  </br>
layers.push_back(new FullConnLayer(8, 32));  </br>
layers.push_back(new SigmoidLayer());  </br>
layers.push_back(new FullConnLayer(32, 3));   </br>
layers.push_back(new SigmoidLayer());  </br>
NetWrapper\<MeanSquareLossLayer\> *ann = new NetWrapper\<MeanSquareLossLayer\>();  </br>
ann->_build_net(layers); </br>
ann->train(); </br>

2. CNN
> std::vector<Layer*> layers; </br>
layers.push_back(new ConvLayer(1, 6, 5, 5, 24, 24)); </br>
layers.push_back(new SigmoidLayer()); </br>
layers.push_back(new PoolingLayer(6, 6, 2, 2, 12, 12)); </br>
layers.push_back(new ConvLayer(6, 6, 5, 5, 8, 8)); </br>
layers.push_back(new SigmoidLayer()); </br>
layers.push_back(new PoolingLayer(6, 6, 2, 2, 4, 4)); </br>
layers.push_back(new ConvLayer(6, 10, 2, 2, 3, 3)); </br>
layers.push_back(new SigmoidLayer()); </br>
layers.push_back(new FlatternLayer()); </br>
layers.push_back(new FullConnLayer(90, 32)); </br>
layers.push_back(new SigmoidLayer()); </br>
layers.push_back(new FullConnLayer(32, 4)); </br>
layers.push_back(new SigmoidLayer()); </br>
NetWrapper\<MeanSquareLossLayer\> *cnn = new NetWrapper\<MeanSquareLossLayer\>(); </br>
cnn->_build_net(layers); </br>
cnn->train();

3. RNN 

3.1 singel layer
> std::vector<Layer*> layers; </br>
layers.push_back(new WordEmbeddingLayer(14)); </br>
layers.push_back(new RnnCell(8, 16)); </br>
layers.push_back(new SeqFullConnLayer(16, 4)); </br>
layers.push_back(new SeqActiveLayer()); </br>
NetWrapper\<SeqLossLayer\> *rnet = new NetWrapper\<SeqLossLayer\>(4); </br>
rnet->_build_net(layers);  </br>

3.2 multi layers
> std::vector<Layer*> layers; </br>
layers.push_back(new WordEmbeddingLayer(14)); </br>
layers.push_back(new RnnCell(8, 8)); </br>
layers.push_back(new RnnCell(8, 16)); </br>
layers.push_back(new SeqFullConnLayer(16, 4)); </br>
layers.push_back(new SeqActiveLayer()); </br>
NetWrapper\<SeqLossLayer\> *rnet = new NetWrapper\<SeqLossLayer\>(4); </br>
rnet->_build_rnn(layers);  </br>

> std::vector<Layer*> layers; </br>
layers.push_back(new WordEmbeddingLayer(14)); </br>
layers.push_back(new LstmCell(8, 8)); </br>
layers.push_back(new LstmCell(8, 16)); </br>
layers.push_back(new SeqFullConnLayer(16, 4)); </br>
layers.push_back(new SeqActiveLayer()); </br>
NetWrapper\<SeqLossLayer\> *rnet = new ReccurentNet\<SeqLossLayer\>(4); </br>
rnet->_build_net(layers);  </br>


3.3 different layers
> std::vector<Layer*> layers; </br>
layers.push_back(new WordEmbeddingLayer(14)); </br>
layers.push_back(new RnnCell(8, 8)); </br>
layers.push_back(new LstmCell(8, 16)); </br>
layers.push_back(new SeqFullConnLayer(16, 4)); </br>
layers.push_back(new SeqActiveLayer()); </br>
NetWrapper\<SeqLossLayer\> *rnet = new NetWrapper\<SeqLossLayer\>(4); </br>
rnet->_build_net(layers);  </br>

3.4 singel bi-directional rnn cell
>std::vector<Layer*> layers; </br>
layers.push_back(new WordEmbeddingLayer(14)); </br>
layers.push_back(new BiCellWrapper\<RnnCell\>(14, 16, BI_RNN_CELL)); </br>
layers.push_back(new SeqFullConnLayer(16, 4)); </br>
layers.push_back(new SeqActiveLayer()); </br>
NetWrapper\<SeqLossLayer\> *rnet = new NetWrapper\<SeqLossLayer\>(4); </br>
rnet->_build_net(layers);  </br>

3.5 multi bi-directional rnn cells
>std::vector<Layer*> layers; </br>
layers.push_back(new WordEmbeddingLayer(14)); </br>
layers.push_back(new BiCellWrapper\<RnnCell\>(14, 16, BI_RNN_CELL)); </br>
layers.push_back(new BiCellWrapper\<LstmCell\>(14, 16, true, true, BI_LSTM_CELL)); </br>
layers.push_back(new SeqFullConnLayer(16, 4)); </br>
layers.push_back(new SeqActiveLayer()); </br>
NetWrapper\<SeqLossLayer\> *rnet = new NetWrapper\<SeqLossLayer\>(4); </br>
rnet->_build_net(layers);  </br>

