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
} pooling_type;

enum {
    FULL = 0,
    SAME,
    VALID
} conv2_type;

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
        for (size_t i = 0; i < _x_dim; i++) {
            for (size_t j = 0; j < _y_dim; j++) {
                _val[i][j] = ( (double)( 2.0 * rand() ) / ((double)RAND_MAX + 1.0) - 1.0 ) ;
            //    uniform_plus_minus_one;
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
    //    return ;
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

    void operator = (int val) const {
        for (int i = 0; i < _x_dim; i++) {
            for (int j = 0; j < _y_dim; j++) {
                _val[i][j] = val;
            }
        }
    }
    
    Matrix<T> operator - (const Matrix<T>& m) const {
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
    
    Matrix<T> operator + (T val) const {
        Matrix<T> t_matrix(_x_dim, _y_dim);
        for (size_t i = 0; i < _x_dim; i++) {
            for (size_t j = 0; j < _y_dim; j++) {
                t_matrix[i][j] = _val[i][j] + val;
            }
        }
        return t_matrix;
    }
    
    Matrix<T> operator - (T val) const {
        Matrix<T> t_matrix(_x_dim, _y_dim);
        for (size_t i = 0; i < _x_dim; i++) {
            for (size_t j = 0; j < _y_dim; j++) {
                t_matrix[i][j] = _val[i][j] - val;
            }
        }
        return t_matrix;
    }
    
    Matrix<T> operator / (T val) const {
        if (val == 0.0) {
            std::cerr << "divided by error!" << std::endl;
            exit(1);
        }
        Matrix<T> t_matrix(_x_dim, _y_dim);
        for (size_t i = 0; i < _x_dim; i++) {
            for (size_t j = 0; j < _y_dim; j++) {
                t_matrix[i][j] = _val[i][j] / val;
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
            std::cerr << "Error when add other a matrix[size not match!]" << std::endl;
            exit(1);
        }
        for (size_t i = 0; i < _x_dim; ++i) {
            for (size_t j = 0; j < _y_dim; ++j) {
                _val[i][j] += m[i][j];
            }
        }
    }

	Matrix<T> minus_by(T val) const {
        Matrix<T> t_matrix;
		for (size_t i = 0; i < _x_dim; ++i) {
            for (size_t j = 0; j < _y_dim; ++j) {
                t_matrix[i][j] = val - _val[i][j];
            }
        }
		return t_matrix;
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
    
    T avg() const {
        if (_x_dim == 0 || _y_dim == 0) {
            std::cerr << "Error when calculate avg of the matrix!" << std::endl;
            exit(1);
        }
        T ret_val = 0.0;
        for (size_t i = 0; i < _x_dim; ++i) {
            for (size_t j = 0; j < _y_dim; ++j) {
                ret_val += _val[i][j];
            }
        }
        return ret_val / (_x_dim * _y_dim);
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

    void resize(T val) {
        for (int i = 0; i < _x_dim; i++) {
            for (int j = 0; j < _y_dim;j ++) {
                _val[i][j] = val;
            }
        }
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

    Matrix<T> local(int r, int c, int rlen, int clen) const {
        if (r + rlen > _x_dim || c + clen > _y_dim) {
            std::cerr << "Error when get local data [ size not match!]" << std::endl;
            exit(1);
        }
        Matrix<T> t_matrix(rlen, clen);
        for (int i = 0; i < rlen; i++) {
            for (int j = 0; j < clen; j++) {
                t_matrix[i][j] = _val[i + r][j + c];
            }
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
        for (int i = 0; i < t_matrix._x_dim; i++) {
            for (int j = 0; j < t_matrix._y_dim; j++) {
                t_matrix[i][j] = (kernel.rotate_180()
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
                t_matrix[i][j] = _val[i / up_x_dim][j / up_y_dim];
            }
        }
        return t_matrix;
    }

    Matrix<T> rotate_180() const {
		Matrix<T> t_matrix(_x_dim, _y_dim);
        for (int i = 0; i < _x_dim * _y_dim; i++) {
			t_matrix[i / _y_dim][i % _y_dim] = 
                _val[_x_dim - 1 - i / _y_dim][_y_dim - 1 - i % _y_dim];
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
                    _val[i][j];
            }
        }
        full_mat._display("full_mat");
        kernel.rotate_180()._display("kernel.rotate_180()");
        dst_mat = full_mat.conv(kernel.rotate_180());
        return dst_mat;
    }

};

typedef Matrix<float> matrix_float;
typedef Matrix<int> matrix_int;
typedef Matrix<double> matrix_double;

}

#endif

