#ifndef _UTIL_H
#define _UTIL_H

#include <iostream>
#include <vector>
#include <math.h>
#include "matrix.h"

namespace sub_dl {

template <typename T>
int merge(const Matrix<T>& output_val) {
    int val = 0;
    int d = 1;
    for (int i = 8; i >= 1; i--) {
        val += int(output_val[i - 1][0] + 0.5) * pow(2, i - 1);
    }
    return val;
}

template <typename T>
std::string merge(const Matrix<T>& output_val, int wordseg) {
    std::string ret = "";
    for (int i = 0; i < output_val._x_dim; i++) {
        int d = 1;
        T val = 0.0;
        for (int j = 0; j < output_val._y_dim; j++) {
            if (val < output_val[i][j]) {
                val = output_val[i][j];
                d = j + 1;
            }
        }
        ret += char('0' + d);
        ret += "_";
    }
    return ret;
}

template <typename T>
void gradient_clip(Matrix<T>& matrix, double clip_gra) {	
	T tot = 0.0;
    for (size_t i = 0; i < matrix._x_dim; i++) {
        for (size_t j = 0; j < matrix._y_dim; j++) {
            tot += (matrix[i][j] * matrix[i][j]);
        }
    }
	T val = sqrt(tot);
	if (val > clip_gra) {
		T scale = clip_gra / val;
		for (int i = 0; i < matrix._x_dim; i++) {
			for (int j = 0; j < matrix._y_dim; j++) {
				matrix[i][j] *= scale;
			}
		}
	}
}

template <typename T>
T tanh(T x) {
	if (x > 400) {
		return 1.0;
	}
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
}

template <typename T>
T sigmoid(T x) {
    return 1.0 / (1 + exp(-x));
}

template <typename T>
T sigmoid_diff(T x) {
    return x * (1 - x);
}

template <typename T>
Matrix<T> tanh_m(const Matrix<T>& matrix) {
    Matrix<T> ret_val(matrix._x_dim, matrix._y_dim);
    for (size_t i = 0; i < matrix._x_dim; i++) {
        for (size_t j = 0; j < matrix._y_dim; j++) {
            ret_val[i][j] = tanh(matrix[i][j]);
        }
    }
    return ret_val;
}

template <typename T>
Matrix<T> tanh_m_diff(const Matrix<T>& matrix) {
    Matrix<T> ret_val(matrix._x_dim, matrix._y_dim);
    for (size_t i = 0; i < matrix._x_dim; i++) {
        for (size_t j = 0; j < matrix._y_dim; j++) {
            ret_val[i][j] = 1 - matrix[i][j] * matrix[i][j];
        }
    }
    return ret_val;
}

template <typename T>
Matrix<T> sigmoid_m(const Matrix<T>& matrix) {
    Matrix<T> ret_val(matrix._x_dim, matrix._y_dim);
    for (size_t i = 0; i < matrix._x_dim; i++) {
        for (size_t j = 0; j < matrix._y_dim; j++) {
            ret_val[i][j] = sigmoid(matrix[i][j]);
        }
    }
    return ret_val;
}

template <typename T>
Matrix<T> sigmoid_m_diff(const Matrix<T>& matrix) {
    Matrix<T> ret_val(matrix._x_dim, matrix._y_dim);
    for (size_t i = 0; i < matrix._x_dim; i++) {
        for (size_t j = 0; j < matrix._y_dim; j++) {
            ret_val[i][j] = matrix[i][j] * (1 - matrix[i][j]);
        }
    }
    return ret_val;
}

template <typename T>
Matrix<T> exp_m(const Matrix<T>& matrix) {
    Matrix<T> ret_val(matrix._x_dim, matrix._y_dim);
    for (size_t i = 0; i < matrix._x_dim; i++) {
        for (size_t j = 0; j < matrix._y_dim; j++) {
            ret_val[i][j] = exp(matrix[i][j]);
        }
    }
    return ret_val;
}

template <typename T>
Matrix<T> log_m(const Matrix<T>& matrix) {
    Matrix<T> ret_val(matrix._x_dim, matrix._y_dim);
    for (size_t i = 0; i < matrix._x_dim; i++) {
        for (size_t j = 0; j < matrix._y_dim; j++) {
            ret_val[i][j] = log(matrix[i][j]);
        }
    }
    return ret_val;
}

}

#endif
