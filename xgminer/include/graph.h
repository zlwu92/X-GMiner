#ifndef __GRAPH_H__
#define __GRAPH_H__

#include <iostream>
#include <vector>
#include <algorithm>
#include "common.h"

enum InputFileType {
    sparse_mtx = 0,
    edge_bin = 1,
    edge_txt = 2,
};


class Graph {
public:
    Graph(std::string file, int input_type);
    ~Graph() {}

    Graph(const Graph &)=delete; // disable copy constructor
    Graph& operator=(const Graph &)=delete; // disable assignment operator

protected:
    std::string name_;            // name of the graph
    std::string inputfile_path;   // file path of the graph
    bool is_directed_;            // is it a directed graph?
    bool is_bipartite;            // is it a bipartite graph?
    bool has_reverse;             // has reverse/incoming edges maintained
    vidType max_degree;           // maximun degree
    vidType n_vertices;           // number of vertices
    eidType n_edges;              // number of edges
    eidType nnz;                  // number of edges in COO format (may be halved due to orientation)

    vidType *colinds;             // column indices of CSR format
    eidType *rowptrs;             // row pointers of CSR format
    vidType *src_list, *dst_list; // source and destination vertices of COO format
    VertexList neigh_count;             // neighbor count of each source vertex in the edgelist
};

#endif