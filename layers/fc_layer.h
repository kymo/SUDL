#include "layer.h"
#include "util.h"

namespace sub_dl {

template <typename T>
class FullConnLayer : public ILayer {

private:
	int _x_dim;		// sample dimention
	int _f_dim;		// feature dimention
	int _o_dim;		// output dimention
	Matrix<T> _x;
	Matrix<T> _y;
	Matrix<float> _w;
	Vector<flaot> _b;

public:
	FullConnLayer() {
	}

	FullConnLayer(int x_dim, int f_dim, int o_dim) 
		: _x_dim(x_dim), _f_dim(f_dim), _o_dim(o_dim) {
		_x.resize(_x_dim, _f_dim);
		_y.resize(_y_dim, _o_dim);
		_w.resize(_f_dim, _o_dim);
	}

	virtual ~FullConnLayer() {
	}

	void _forward() {
		

		
	}
	void _backward() {
	
	}
};

}
