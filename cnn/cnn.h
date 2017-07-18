
#ifndef CNN_H_
#define CNN_H_

#include <iostream>
#include "matrix.h"

class CNN {

private:
	std::vector<Layer>* _layers;

	void build_cnn(std::vector<Layer*> layers) {
		_layers = layers;
	}

	void _forward(std::vector<matrix_double> data) {
		
		for (auto layer : layers) {
			// data = layer->_forward(data);
		}
	}


	void train() {		

	}
};


#endif

