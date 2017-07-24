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
> layers.push_back(new FullConnLayer(4, 8));
layers.push_back(new ReluLayer());
layers.push_back(new FullConnLayer(8, 32));
layers.push_back(new SigmoidLayer());
layers.push_back(new FullConnLayer(32, 3)); 
layers.push_back(new SigmoidLayer());
ANN<MeanSquareLossLayer> *ann = new ANN<MeanSquareLossLayer>();
ann->build_ann(layers);
ann->load_data(argv[1]);
ann->train();

