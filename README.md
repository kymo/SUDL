# SUDL

A light deep learning tools box by c++

## Contains

**Network Architecture**
1. Convolutional Neural Network 
2. Normal Neural Network
3. Reccurent Neural Network with three mainstream varieties(LSTM, LSTM-peelhole, GRU)

**Nonlinearities**
1. ReLU
2. Sigmoid
3. tanh

**Usage**
1. ANN
> layers.push_back(new FullConnLayer(4, 8));  </br>
layers.push_back(new ReluLayer());  </br>
layers.push_back(new FullConnLayer(8, 32));  </br>
layers.push_back(new SigmoidLayer());  </br>
layers.push_back(new FullConnLayer(32, 3));   </br>
layers.push_back(new SigmoidLayer());  </br>
ANN<MeanSquareLossLayer> *ann = new ANN<MeanSquareLossLayer>();  </br>
ann->build_ann(layers); </br>
ann->load_data(argv[1]); </br>
ann->train(); </br>

2. CNN
> std::vector<Layer*> layers; </br>
layers.push_back(new ConvLayer(1, 6, 5, 5, 24, 24)); </br>
layers.push_back(new SigmoidLayer()); </br>
layers.push_back(new PoolingLayer(6, 6, 2, 2, 12, 12)); </br>
layers.push_back(new SigmoidLayer()); </br>
layers.push_back(new ConvLayer(6, 6, 5, 5, 8, 8)); </br>
layers.push_back(new SigmoidLayer()); </br>
layers.push_back(new PoolingLayer(6, 6, 2, 2, 4, 4)); </br>
layers.push_back(new SigmoidLayer()); </br>
layers.push_back(new ConvLayer(6, 10, 2, 2, 3, 3)); </br>
layers.push_back(new SigmoidLayer()); </br>
layers.push_back(new FlatternLayer()); </br>
layers.push_back(new FullConnLayer(90, 32)); </br>
layers.push_back(new SigmoidLayer()); </br>
layers.push_back(new FullConnLayer(32, 4)); </br>
layers.push_back(new SigmoidLayer()); </br>
CNN<MeanSquareLossLayer> *cnn = new CNN<MeanSquareLossLayer>(); </br>
cnn->build_cnn(layers); </br>
cnn->load_data(argv[1]); </br>
cnn->train();

