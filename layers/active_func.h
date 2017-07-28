// Copyright (c) 2017 kymowind@gmail.com. All Rights Reserve.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//    http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License. 

#ifndef ACTIVE_FUNC_H
#define ACTIVE_FUNC_H

#include <iostream>
#include <string>
#include <algorithm>
#include "util.h"

namespace sub_dl {

enum {
    SIGMOID = 0,        // sigmoid
    RELU,                // relu
    TANH,                // tanh
    LEAKY_RELU            // leaky relu
};

template <typename T>
class ActiveFunc {
public:
    virtual Matrix<T> _calc(const Matrix<T>& matrix) = 0;
    virtual Matrix<T> _diff(const Matrix<T>& matrix) = 0;
};

template <typename T>
class Sigmoid : public ActiveFunc<T> {
public:
    // sigmoid calculation
    Matrix<T> _calc(const Matrix<T>& matrix) {
        return sigmoid_m(matrix);
    }
    
    // sigmoid diff
    Matrix<T> _diff(const Matrix<T>& matrix) {
        return sigmoid_m_diff(sigmoid_m(matrix));
    }
};

class LeakyRelu {
    
};

template <typename T>
class Relu : public ActiveFunc<T>{
public:

    // relu calculation
    Matrix<T> _calc(const Matrix<T>& matrix) {
        Matrix<T> t_matrix(matrix._x_dim, matrix._y_dim);
        for (int i = 0; i < matrix._x_dim; i++) {
            for (int j = 0; j < matrix._y_dim; j++) {
                t_matrix[i][j] = matrix[i][j] > 0 ? matrix[i][j] : 0;
            }
        }
        return t_matrix;
    }
    
    // relu differential
    Matrix<T> _diff(const Matrix<T>& matrix) {
        Matrix<T> t_matrix(matrix._x_dim, matrix._y_dim);
        for (int i = 0; i < matrix._x_dim; i++) {
            for (int j = 0; j < matrix._y_dim; j++) {
                t_matrix[i][j] = matrix[i][j] > 0 ? 1 : 0;
            }
        }
        return t_matrix;
    }
};

template <typename T>
class Tanh : public ActiveFunc<T> {
public:
    Matrix<T> _calc(const Matrix<T>&matrix) {
        return tanh_m(matrix);
    }
    
    Matrix<T> _diff(const Matrix<T>& matrix) {
        return tanh_m_diff(tanh_m(matrix));
    }
};

template <typename T>
class ActiveFuncFactory {

private:
    ActiveFuncFactory() {}
    static ActiveFuncFactory<T>* _instance;

public:
    
    static ActiveFuncFactory<T>* _get_instance() {
        if (NULL == _instance) {
            _instance = new ActiveFuncFactory<T>();
        }
        return _instance;
    }

    // create the active func according to the func type
    ActiveFunc<T>* _produce(int active_func_type) {
        switch(active_func_type) {
        case SIGMOID:
            return new Sigmoid<T>();
            break;
        case RELU:
            return new Relu<T>();
            break;
        case TANH:
            return new Tanh<T>();
            break;
        }
    }
};

template <typename T>
ActiveFuncFactory<T>* ActiveFuncFactory<T>::_instance = NULL;

}
#endif
