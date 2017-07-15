#ifndef _UTIL_H
#define _UTIL_H

#include <iostream>
#include <vector>
#include "matrix.h"

namespace sub_dl {
template <typename T>
void gradident_clip(Matrix<T>& matrix, double clip_gra) {	
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
            ret_val[i][j] = 1 - tanh(matrix[i][j]) * tanh(matrix[i][j]);
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

}

#endif
