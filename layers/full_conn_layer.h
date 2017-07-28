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

#ifndef FULLCONN_LAYER_H
#define FULLCONN_LAYER_H

#include "layer.h"
#include "base_full_conn_layer.h"

namespace sub_dl {

class FullConnLayer : public BaseFullConnLayer {

public:

    virtual ~FullConnLayer() {}

    FullConnLayer(int input_dim, int output_dim) :
        BaseFullConnLayer(input_dim, output_dim) {
    }
    void _forward(Layer* pre_layer);
    void _backward(Layer* nxt_layer);

};

}
#endif
