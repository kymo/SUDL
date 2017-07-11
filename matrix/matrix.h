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
void destroy_matrix(T** _val, int r) {
    for (int i = 0; i < r; i ++) {
        if (NULL != _val[i]) {
            delete [] _val[i];
        }
    }
}

template <typename T>
class Matrix {

private:
    T** _val;

public:
    int _x_dim;        // x dimention of matrix
    int _y_dim;        // y dimention of matrix
    
    Matrix() {
        _val = NULL;
        _x_dim = 0;
        _y_dim = 0;
    }

    virtual~ Matrix() {
        destroy_matrix<T>(_val, _x_dim);
    }

    void destroy() {
        destroy_matrix<T>(_val, _x_dim);
    }

    void assign_val() {
        srand((unsigned) time(NULL));
        for (size_t i = 0; i < _x_dim; i++) {
            for (size_t j = 0; j < _y_dim; j++) {
                _val[i][j] = (rand() % 100) / 1000.0;
            }
        }
    }

    Matrix(int x_dim, int y_dim) {
        _x_dim = x_dim;
        _y_dim = y_dim;
        alloc_matrix<T>(_val, _x_dim, _y_dim);
    }

    void operator = (const Matrix<T>& m) {
        _x_dim = m._x_dim;
        _y_dim = m._y_dim;
        alloc_matrix<T>(_val, m._x_dim, m._y_dim);
        for (size_t i = 0; i < _x_dim; i++) {
            for (size_t j = 0; j < _y_dim; j++) {
                _val[i][j] = m[i][j];
            }
        }    
    }

    Matrix(const Matrix<T>& m) {
        _val = NULL;
        _x_dim = m._x_dim;
        _y_dim = m._y_dim;
        alloc_matrix<T>(_val, m._x_dim, m._y_dim);
        for (size_t i = 0; i < _x_dim; i++) {
            for (size_t j = 0; j < _y_dim; j++) {
                _val[i][j] = m[i][j];
            }
        }
    }
    
    void _display(const std::string& tips) const {
        std::cout << tips << std::endl;
        _display();
    }

    void _display() const {
        for (size_t i = 0; i < _x_dim; i++) {
            for (size_t j = 0; j < _y_dim; j++) {
                std::cout << _val[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }

    // operator inherit
    T* operator [] (int i) const {
        return _val[i];
    }

    // operator inherit
    Matrix<T> operator + (const Matrix<T>& m) {
        if (m._x_dim != _x_dim || m._y_dim != _y_dim) {
            std::cerr << "Error when add two matrix[size not match!]" << _x_dim << " " << _y_dim << 
                " vs " << m._x_dim << " " << m._y_dim << std::endl;
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
        if (m._x_dim != _x_dim || m._y_dim != _y_dim) {
            std::cerr << "Error when minus two matrix[size not match!]" << _x_dim << " " << _y_dim << 
                " vs " << m._x_dim << " " << m._y_dim << std::endl;
            exit(1);
        }
        Matrix<T> t_matrix(_x_dim, _y_dim);
        for (size_t i = 0; i < _x_dim; ++i) {
            for (size_t j = 0; j < _y_dim; ++j) {
                t_matrix[i][j] = _val[i][j] - m[i][j];
            }
        }
        return t_matrix;
    }

    Matrix<T> operator * (float rate) const {
        Matrix<T> t_matrix(_x_dim, _y_dim);
        for (size_t i = 0; i < _x_dim; i++) {
            for (size_t j = 0; j < _y_dim; j++) {
                t_matrix[i][j] = _val[i][j] * rate;
            }
        }
        return t_matrix;
    }

    Matrix<T> operator * (const Matrix<T>& m) const {
        if (m._x_dim != _y_dim) {
            std::cerr << "Error when multiply two matrix[size not match!]" << _x_dim << " " << _y_dim << 
                " vs " << m._x_dim << " " << m._y_dim << std::endl;
            exit(1);
        }
        Matrix<T> t_matrix(_x_dim, m._y_dim);
        for (size_t i = 0; i < _x_dim; i++) {
            for (size_t j = 0; j < m._y_dim; j++) {
                T tot_val = 0;
                for (size_t k = 0; k < _y_dim; k++) {
                    tot_val += _val[i][k] * m[k][j];
                }
                t_matrix[i][j] = tot_val;
            }
        }
        return t_matrix;
    }

    Matrix<T> dot_mul(const Matrix<T>& m) const {
        if (m._x_dim != _x_dim || m._y_dim != _y_dim) {
            std::cerr << "Error when dot multiply two matrix[size not match!]" << std::endl;
            exit(1);
        }
        Matrix<T> t_matrix(_x_dim, _y_dim);
        for (size_t i = 0; i < _x_dim; ++i) {
            for (size_t j = 0; j < _y_dim; ++j) {
                t_matrix[i][j] = _val[i][j] * m[i][j];
            }
        }
        return t_matrix;
    }

    void add(const Matrix<T>& m) const {
        if (m._x_dim != _x_dim || m._y_dim != _y_dim) {
            std::cerr << "Error when dot multiply two matrix[size not match!]" << std::endl;
            exit(1);
        }
        for (size_t i = 0; i < _x_dim; ++i) {
            for (size_t j = 0; j < _y_dim; ++j) {
                _val[i][j] += m[i][j];
            }
        }
    }
    
    T sum() const {
        T ret_val = 0.0;
        for (size_t i = 0; i < _x_dim; ++i) {
            for (size_t j = 0; j < _y_dim; ++j) {
                ret_val += _val[i][j];
            }
        }
        return ret_val;
    }

    Matrix<T> _R(int r) const {
        if (_x_dim <= r) {
            std::cerr << "Error when get row [" << r << "] in matrix" << std::endl;
            exit(0);
        }
        Matrix<T> ret_val(1, _y_dim);
        for (size_t i = 0; i < _y_dim; i++) {
            ret_val[0][i] = _val[r][i];
        }
        return ret_val;
    }

    Matrix<T> _C(int c) const {
        if (_y_dim <= c) {
            std::cerr << "Error when get col [" << c << "] in matrix" << std::endl;
            exit(0);
        }
        Matrix<T> ret_val(_x_dim, 1);
        for (size_t i = 0; i < _x_dim; i++) {
            ret_val[i][0] = _val[i][c];
        }
    }
    
    /*
    * @brief set row of the values
    * @param r row number in current instance
    * @param the values that needs to be setted, and its row number should be 1
    */
    void set_row(int r, const Matrix<T>& matrix, int src_row = 0) { 
        if (matrix._x_dim != 1) {
            std::cerr << "Error when set row in matrix" << std::endl;
            exit(0);
        }
        for (size_t i = 0; i < _y_dim; i++) {
            _val[r][i] = matrix[src_row][i];
        }
    }

    void resize(int x_dim, int y_dim) {
        _x_dim = x_dim;
        _y_dim = y_dim;
        alloc_matrix<T>(_val, _x_dim, _y_dim);
    }
    
    Matrix<T> _T() const {
        Matrix<T> t_matrix(_y_dim, _x_dim);
        for (size_t i = 0; i != _y_dim; ++i) {
            for (size_t j = 0; j != _x_dim; ++j) {
                t_matrix[i][j] = _val[j][i];
            }
        }
        return t_matrix;
    }
};

typedef Matrix<float> matrix_float;
typedef Matrix<int> matrix_int;

}

#endif

