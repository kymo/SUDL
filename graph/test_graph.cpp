
#include "graph.h"
#include "embedding_layer.h"
#include "seq_full_conn_softmax_layer.h"

#include <stdarg.h> 

using namespace sub_dl;

std::vector<int> REFER(int n, ...) {
    va_list args;
    va_start(args, n);
    std::vector<int> ret;
    while (n > 0) {
        ret.push_back(va_arg(args, int));
        n --;
    }
    return ret;
}

int main() {
    
    Graph *graph = new Graph();

    int emb_id = graph->_add_node(new WordEmbeddingLayer(14), REFER(1, 0));
    
    int lstm_id = graph->_add_node(new LstmCell(14, 16, true), REFER(1, emb_id)); 
    int seq_full_id = graph->_add_node(new SeqFullConnSoftmaxLayer(16, 4), REFER(1, lstm_id));
    
    int seq_full_id1 = graph->_add_node(new SeqFullConnSoftmaxLayer(16, 4), REFER(2, emb_id, seq_full_id));

    graph->_forward_compute();

    return 0;
}
