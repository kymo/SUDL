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

#ifndef MATRIX_H_
#define MATRIX_H_

#include <iostream>
#include <string>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <time.h>
#include <math.h>

namespace sub_dl {

#define uniform_plus_minus_one ( (double)( 2.0 * rand() ) / ((double)RAND_MAX + 1.0) - 1.0 ) 

enum {
    AVG_POOLING = 0,
    MAX_POOLING
};

enum {
    FULL = 0,
    SAME,
    VALID
};

template <typename T>
void alloc_matrix(T* &_val, int r, int c) {
    _val = new T[r * c];
    for (size_t i = 0; i < r * c; i ++) {
        _val[i] = 0;
    }
}

template <typename T>
void destroy_matrix(T* _val) {
    delete[] _val;
    _val = NULL;
}

template <typename T>
class Matrix {

public:
    T* _val;
    
    int _x_dim;        // x dimention of matrix
    int _y_dim;        // y dimention of matrix
 
    Matrix() {
        _val = NULL;
        _x_dim = 0;
        _y_dim = 0;
        
    }

    ~ Matrix() {
        destroy_matrix<T>(_val);
    }

    void destroy() {
        destroy_matrix<T>(_val, _x_dim);
    }

    void assign_val() {
        for (int i = 0; i < _x_dim * _y_dim; i++) {
            _val[i] = ( (double)( 2.0 * rand() ) / ((double)RAND_MAX + 1.0) - 1.0 ) ;
        }
    }

    Matrix(int x_dim, int y_dim) {
        _x_dim = x_dim;
        _y_dim = y_dim;
        alloc_matrix<T>(_val, _x_dim, _y_dim);
    }

    void operator = (const Matrix<T>& m) {
        if (_x_dim != 0 && _x_dim != m._x_dim) {
            std::cerr << "Error whenn operator = !" << std::endl;
            exit(1);
        }
        if (_x_dim == 0) {
            _x_dim = m._x_dim;
            _y_dim = m._y_dim;
            alloc_matrix<T>(_val, m._x_dim, m._y_dim);
        }

        for (int i = 0; i < _x_dim * _y_dim; i++) {
            _val[i] = m._val[i];
        }
    }

    Matrix(const Matrix<T>& m) {
        _val = NULL;
        _x_dim = m._x_dim;
        _y_dim = m._y_dim;
        alloc_matrix<T>(_val, m._x_dim, m._y_dim);
        for (int i = 0; i < _x_dim * _y_dim; i++) {
            _val[i] = m._val[i];
        }
    }
    
    void _display(const std::string& tips) const {
        std::cout << tips << std::endl;
        _display();
    }

    void _display() const {
        for (size_t i = 0; i < _x_dim; i++) {
            for (size_t j = 0; j < _y_dim; j++) {
                std::cout << _val[i * _y_dim + j] << " ";
            }
            std::cout << std::endl;
        }
    }

    // operator inherit
    T* operator [] (int i) const {
        return _val + i * _y_dim;
    }

    // operator inherit
    Matrix<T> operator + (const Matrix<T>& m) {
        if (m._x_dim != _x_dim || m._y_dim != _y_dim) {
            std::cerr << "Error when add two matrix[size not match!]" << _x_dim << " " << _y_dim << 
                " vs " << m._x_dim << " " << m._y_dim << std::endl;
            exit(1);
        }
        Matrix<T> t_matrix(_x_dim, _y_dim);
        for (int i = 0; i < _x_dim * _y_dim; i++) {
            t_matrix._val[i] = m._val[i] + _val[i];
        }
        return t_matrix;
    }

    void operator = (int val) const {
        for (int i = 0; i < _x_dim * _y_dim; i++) {
            _val[i] = val;
        }
    }
    
    Matrix<T> operator - (const Matrix<T>& m) const {
        if (m._x_dim != _x_dim || m._y_dim != _y_dim) {
            std::cerr << "Error when minus two matrix[size not match!]" << _x_dim << " " << _y_dim << 
                " vs " << m._x_dim << " " << m._y_dim << std::endl;
            exit(1);
        }
        Matrix<T> t_matrix(_x_dim, _y_dim);
        for (size_t i = 0; i < _x_dim * _y_dim; ++i) {
            t_matrix._val[i] = _val[i] - m._val[i];
        }
        return t_matrix;
    }

    Matrix<T> operator * (float rate) const {
        Matrix<T> t_matrix(_x_dim, _y_dim);
        for (size_t i = 0; i < _x_dim * _y_dim; i++) {
            t_matrix._val[i] = _val[i] * rate;
        }
        return t_matrix;
    }
    
    Matrix<T> operator + (T val) const {
        Matrix<T> t_matrix(_x_dim, _y_dim);
        for (size_t i = 0; i < _x_dim * _y_dim; i++) {
            t_matrix._val[i] = _val[i] + val;
        }
        return t_matrix;
    }
    
    Matrix<T> operator - (T val) const {
        Matrix<T> t_matrix(_x_dim, _y_dim);
        for (size_t i = 0; i < _x_dim * _y_dim; i++) {
            t_matrix._val[i] = _val[i] - val;
        }
        return t_matrix;
    }
    
    Matrix<T> operator / (T val) const {
        if (val == 0.0) {
            std::cerr << "divided by error!" << std::endl;
            exit(1);
        }
        Matrix<T> t_matrix(_x_dim, _y_dim);
        for (size_t i = 0; i < _x_dim * _y_dim; i++) {
            t_matrix._val[i] = _val[i] / val;
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
        //cblas_sgemm(Order, TransA, TransB, _x_dim, m._y_dim, _y_dim, 1, 
        //    _val[0], _y_dim, m[0], m._y_dim, 0,  t_matrix[0], t_matrix._y_dim);
        for (int i = 0; i < _x_dim * m._y_dim; i++) {
            T tot_val = 0;
            int ni = i / m._y_dim;
            int nj = i % m._y_dim;
            for (int k = 0; k < _y_dim; k++) {
                tot_val += _val[ni * _y_dim + k] * m._val[k * m._y_dim + nj];
            }
            t_matrix._val[i] = tot_val;
        }
        /*

        for (size_t i = 0; i < _x_dim; i++) {
            for (size_t j = 0; j < m._y_dim; j++) {
                T tot_val = 0;
                for (size_t k = 0; k < _y_dim; k++) {
                    tot_val += _val[i][k] * m[k][j];
                }
                t_matrix[i][j] = tot_val;
            }
        }
        */

        return t_matrix;
    }

    /*

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
    */

    Matrix<T> dot_mul(const Matrix<T>& m) const {
        if (m._x_dim != _x_dim || m._y_dim != _y_dim) {
            std::cerr << "Error when dot multiply two matrix[size not match!]" << std::endl;
            exit(1);
        }
        Matrix<T> t_matrix(_x_dim, _y_dim);
        for (int i = 0; i < _x_dim * _y_dim; i++) {
            t_matrix._val[i] = _val[i] * m._val[i];
        }
        return t_matrix;
    }

    void add(const Matrix<T>& m) const {
        if (m._x_dim != _x_dim || m._y_dim != _y_dim) {
            std::cerr << "Error when add other a matrix[size not match!]" << std::endl;
            exit(1);
        }
        for (size_t i = 0; i < _x_dim * _y_dim; ++i) {
            _val[i] = _val[i] + m._val[i];
        }
    }

    Matrix<T> minus_by(T val) const {
        Matrix<T> t_matrix;
        for (size_t i = 0; i < _x_dim * _y_dim; ++i) {
            t_matrix._val[i] = val - _val[i];
        }
        return t_matrix;
    }
    
    T sum() const {
        T ret_val = 0.0;
        for (size_t i = 0; i < _x_dim * _y_dim; ++i) {
            ret_val += _val[i];
        }
        return ret_val;
    }
    
    T avg() const {
        if (_x_dim == 0 || _y_dim == 0) {
            std::cerr << "Error when calculate avg of the matrix!" << std::endl;
            exit(1);
        }
        return sum() / (_x_dim * _y_dim);
    }

    Matrix<T> _R(int r) const {
        if (_x_dim <= r) {
            std::cerr << "Error when get row [" << r << "] in matrix" << std::endl;
            exit(0);
        }
        Matrix<T> ret_val(1, _y_dim);
        for (size_t i = 0; i < _y_dim; i++) {
            ret_val._val[i] = _val[r * _y_dim + i];
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
            ret_val._val[i] = _val[i * _y_dim + c];
        }
        return ret_val;
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
        for (int i = 0; i < _y_dim; i++) {
            _val[r * _y_dim + i] = matrix._val[src_row * _y_dim + i];
        }
    }

    void resize(int x_dim, int y_dim) {
        if (_x_dim != 0) {
            destroy_matrix<T>(_val);
        }
        _x_dim = x_dim;
        _y_dim = y_dim;
        alloc_matrix<T>(_val, _x_dim, _y_dim);
    }

    void resize(T val) {
        for (int i = 0; i < _x_dim * _y_dim; i++) {
            _val[i] = val;
        }
    }
    
    Matrix<T> _T() const {
        Matrix<T> t_matrix(_y_dim, _x_dim);
        for (int i = 0; i < _x_dim * _y_dim; i++) {
            int ni = i / _y_dim;
            int nj = i % _y_dim;
            t_matrix._val[nj * _x_dim + ni] = 
                _val[i];
        }
        return t_matrix;
    }
    
    Matrix<T> local(int r, int c, int rlen, int clen) const {
        if (r + rlen > _x_dim || c + clen > _y_dim) {
            std::cerr << "Error when get local data [ size not match!]" << std::endl;
            exit(1);
        }
        Matrix<T> t_matrix(rlen, clen);
        for (int i = 0; i < rlen * clen; i++) {
            int ni = i / clen;
            int nj = i % clen;
            t_matrix._val[i] = 
                _val[(ni + r) * _y_dim + nj + c];
        }
        return t_matrix;
    }

    // for CNN
    Matrix<T> conv(const Matrix<T>& kernel) const {
        if (kernel._x_dim > _x_dim || kernel._y_dim > _y_dim) {
            std::cerr << "Error when conv [size not match!]" << std::endl;
            exit(1);
        }
        Matrix<T> t_matrix(_x_dim - kernel._x_dim + 1,
            _y_dim - kernel._y_dim + 1);
        Matrix<T> kernel_rotated = kernel.rotate_180();
        for (int i = 0; i < t_matrix._x_dim; i++) {
            for (int j = 0; j < t_matrix._y_dim; j++) {
                t_matrix[i][j] = (kernel_rotated
                    .dot_mul(local(i, j, kernel._x_dim, kernel._y_dim))).sum();
            }
        }
        return t_matrix;
    }
    // for pooling
    Matrix<T> down_sample(int pooling_x_dim, int pooling_y_dim, int sample_type) {
        if (_x_dim % pooling_x_dim != 0 || _y_dim % pooling_y_dim != 0) {
            std::cerr << "Error when down_sample [size not match!]" << std::endl;
            exit(1);
        }
        Matrix<T> t_matrix(_x_dim / pooling_x_dim, 
            _y_dim / pooling_y_dim);
        for (int i = 0; i < t_matrix._x_dim; i++) {
            for (int j = 0; j < t_matrix._y_dim; j++) {
                if (sample_type == AVG_POOLING) {
                    t_matrix[i][j] = local(i * pooling_x_dim, 
                        j * pooling_y_dim, 
                        pooling_x_dim, 
                        pooling_y_dim).avg();
                }
            }
        }
        return t_matrix;
    }

    // up  sampling
    Matrix<T> up_sample(int up_x_dim, int up_y_dim) const {
        Matrix<T> t_matrix(_x_dim * up_x_dim, _y_dim * up_y_dim);
        for (int i = 0; i < t_matrix._x_dim; i++) {
            for (int j = 0; j < t_matrix._y_dim; j++) {
                t_matrix[i][j] = _val[_y_dim * (i / up_x_dim) + j / up_y_dim];
            }
        }
        return t_matrix;
    }

    Matrix<T> rotate_180() const {
        Matrix<T> t_matrix(_x_dim, _y_dim);
        for (int i = 0; i < _x_dim * _y_dim; i++) {
            t_matrix._val[i] = 
                _val[_y_dim * (_x_dim - 1 - i / _y_dim) + _y_dim - 1 - i % _y_dim];
        }
        return t_matrix;
    }

    // conv2d just like what the matlab do
    Matrix<T> conv2d(const Matrix<T>& kernel, int shape) const {
        Matrix<T> dst_mat(_x_dim + kernel._x_dim - 1,
            _y_dim + kernel._y_dim - 1);
        Matrix<T> full_mat(_x_dim + 2 * kernel._x_dim - 2,
            _y_dim + 2 * kernel._y_dim - 2);
        for (int i = 0; i < _x_dim; i++) {
            for (int j = 0; j < _y_dim; j++) {
                full_mat[i + kernel._x_dim - 1][j + kernel._y_dim - 1] = 
                    _val[i * _y_dim + j];
            }
        }
        dst_mat = full_mat.conv(kernel);
        return dst_mat;
    }
};

typedef Matrix<float> matrix_float;
typedef Matrix<int> matrix_int;
typedef Matrix<double> matrix_double;

}

#endif

