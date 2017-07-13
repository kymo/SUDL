/*
* file_name : test_matrix.cpp
*/

#include <iostream>

#include "matrix.h"

using namespace std;
using namespace sub_dl;

void test_basic_func() {	
	int **v;
	alloc_matrix<int>(v, 4, 4);
	for (int i = 0; i < 4; i ++) {
		for (int j = 0; j < 4; j ++) {
			std::cout << v[i][j] << " ";
		}
		std::cout << std::endl;
	}
	int *s;
	alloc_vector<int>(s, 4);
	for (int i = 0; i < 4; i ++) {
		std::cout << s[i] << " ";
	}
	std::cout << std::endl;
}

void test_matrix() {
	srand( (unsigned)time( NULL ) );
	Matrix<int> t(4, 4);
	std::cout << "Firat Matrix" << std::endl;
	for (int i = 0; i < 4; i ++) {
		for (int j = 0; j < 4; j ++) {
			t[i][j] = rand() % 4;
			std::cout << t[i][j] << " ";
		}
		std::cout << std::endl;
	}
	Matrix<int> s(4, 4);
	std::cout << "Second Matrix" << std::endl;
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j ++) {
			s[i][j] = rand() % 4;
			std::cout << s[i][j] << " ";
		}
		std::cout << std::endl;
	}
	Matrix<int> out = s + t;	
	std::cout << "Add result" << std::endl;
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j ++) {
			std::cout << out[i][j] << " ";
		}
		std::cout << std::endl;
	}
	Matrix<int>out1 = t - s;
	std::cout << "Minus result" << std::endl;
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j ++) {
			std::cout << out1[i][j] << " ";
		}
		std::cout << std::endl;
	}
	Matrix<int>out2 = (t * s);
	std::cout << "Multiply result" << std::endl;
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j ++) {
			std::cout << out2[i][j] << " ";
		}
		std::cout << std::endl;
	}
	Matrix<int>out3 = (out + t * s);
	std::cout << "Multiply1 result" << std::endl;
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j ++) {
			std::cout << out3[i][j] << " ";
		}
		std::cout << std::endl;
	}
	Matrix<int>out4 = t * s + out;
	std::cout << "Multiply2 result" << std::endl;
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j ++) {
			std::cout << out4[i][j] << " ";
		}
		std::cout << std::endl;
	}
}
int main() {
	//test_basic_func();
	test_matrix();
	return 0;
}
