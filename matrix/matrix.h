#ifndef MATRIX_H_
#define MATRIX_H_

#include <iostream>
#include <string>
#include <stdlib.h>
#include <stdio.h>
#include <vector>

namespace sub_dl {

template <typename T>
void alloc_matrix(T** &_val, int r, int c) {
    _val = new T*[r];
    for (size_t i = 0; i < r; i ++) {
        _val[i] = new T[c];
        for (size_t j = 0; j < c; ++j) {
            _val[i][j] = 0;
        }
    }
}

template <typename T>
void alloc_vector(T* &_val, int r) {
    _val = new T[r];
    for (size_t i = 0; i < r; i++) {
        _val[i] = 0;
    }
}

template <typename T>
void destroy_matrix(T** &_val, int r) {
    for (int i = 0; i < r; i ++) {
        if (NULL != _val[i]) {
            delete [] _val[i];
        }
    }
    delete _val;
}

template <typename T>
void destroy_vector(T* &_val) {
    if (_val != NULL) {
        delete [] _val;
    }
}

template <typename T>
class Matrix {

private:
    T** _val;
    int _x_dim;        // x dimention of matrix
    int _y_dim;        // y dimention of matrix

public:
    Matrix() {
        _val = NULL;
        _x_dim = 0;
        _y_dim = 0;
    }
    virtual~ Matrix() {
        destroy_matrix<T>(_val, _x_dim);
    }

    Matrix(int x_dim, int y_dim) {
        _x_dim = x_dim;
        _y_dim = y_dim;
        alloc_matrix<T>(_val, _x_dim, _y_dim);
    }

    // operator inherit
    T* operator [] (int i) const {
        return _val[i];
    }

    // operator inherit
    Matrix<T> operator + (const Matrix<T>& m) {
        if (m.get_xdim() != get_xdim() || m.get_ydim() != get_ydim()) {
            std::cerr << "Error when add two matrix[size not match!]" << std::endl;
            exit(1);
        }
        Matrix<T> t_matrix(_x_dim, _y_dim);
        for (size_t i = 0; i < _x_dim; ++i) {
            for (size_t j = 0; j < _y_dim; ++j) {
                t_matrix[i][j] = m[i][j] + _val[i][j];
            }
        }
        return t_matrix;
    }
    
    Matrix<T> operator - (const Matrix<T>& m) {
        if (m.get_xdim() != get_xdim() || m.get_ydim() != get_ydim()) {
            std::cerr << "Error when minus two matrix[size not match!]" << std::endl;
            exit(1);
        }
        Matrix<T> t_matrix(_x_dim, _y_dim);
        for (size_t i = 0; i < _x_dim; ++i) {
            for (size_t j = 0; j < _y_dim; ++j) {
                t_matrix[i][j] = m[i][j] - _val[i][j];
            }
        }
        return t_matrix;
    }
    
    Matrix<T> operator * (const Matrix<T>& m) {
        if (m.get_xdim() != get_ydim()) {
            std::cerr << "Error when multiply two matrix[size not match!]" << std::endl;
            exit(1);
        }
        Matrix<T> t_matrix(_x_dim, m.get_ydim());
        for (size_t i = 0; i < _x_dim; i++) {
            for (size_t j = 0; j < m.get_ydim(); j++) {
                T tot_val = 0;
                for (size_t k = 0; k < _y_dim; k++) {
                    tot_val += _val[i][k] * m[k][j];
                }
                t_matrix[i][j] = tot_val;
            }
        }
        return t_matrix;
    }

    const int get_xdim() const {
        return _x_dim;
    }
    
    const int get_ydim() const {
        return _y_dim;
    }

    void resize(int x_dim, int y_dim) {
        _x_dim = x_dim;
        _y_dim = y_dim;
        alloc_matrix<T>(_val, _x_dim, _y_dim);
    }
    
    Matrix<T> _T() {
        Matrix<T> t_matrix(_y_dim, _x_dim);
        for (size_t i = 0; i != _y_dim; ++i) {
            for (size_t j = 0; j != _x_dim; ++j) {
                t_matrix[i][j] = _val[i][j];
            }
        }
        return t_matrix;
    }
};

}

#endif

