#ifndef ACTIVE_FUNC_H
#define ACTIVE_FUNC_H

#include "util.h"
#include <iostream>
#include <string>
#include <algorithm>

namespace sub_dl {

enum {
    SIGMOID = 0,
    RELU,
    LEAKY_RELU
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
    Matrix<T> _calc(const Matrix<T>& matrix) {
        return sigmoid_m(matrix);
    }
    
    Matrix<T> _diff(const Matrix<T>& matrix) {
        return sigmoid_m_diff(sigmoid_m(matrix));
    }
};

class LeakyRelu {
    
};

template <typename T>
class Relu : public ActiveFunc<T>{
public:
    Matrix<T> _calc(const Matrix<T>& matrix) {
        Matrix<T> t_matrix(matrix._x_dim, matrix._y_dim);
        for (int i = 0; i < matrix._x_dim; i++) {
            for (int j = 0; j < matrix._y_dim; j++) {
                t_matrix[i][j] = matrix[i][j] > 0 ? matrix[i][j] : 0;
            }
        }
        return t_matrix;
    }
    
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

    ActiveFunc<T>* _produce(int active_func_type) {
        switch(active_func_type) {
        case SIGMOID:
            return new Sigmoid<T>();
            break;
        case RELU:
            return new Relu<T>();
            break;
        }
    }
};

template <typename T>
ActiveFuncFactory<T>* ActiveFuncFactory<T>::_instance = NULL;

}

#endif
