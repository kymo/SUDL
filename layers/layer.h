
#ifndef _LAYER_H_
#define _LAYER_H_

#include <iostream>
#include <vector>

namespace sub_dl {

class ILayer {

public:
	virtual ~ILayer() {
	
	}
	ILayer() {
	
	}

	virtual void _forward() = 0;
	virtual void _backward() = 0;

};

}


#endif
