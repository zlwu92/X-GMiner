/*
    * graph.h
    *
    * borrowed from GraphPi and GLUMIN
*/

#pragma once
#include <iostream>
#include <vector>
#include <algorithm>
#include "common.h"
#include "schedule.h"
#include "vertex_set.h"
#include <assert.h>

class Graphmpi;

enum InputFileType {
    sparse_mtx = 0,
    edge_bin = 1,
    edge_txt = 2,
};

class Graph {
public:
    /********************************** GraphMiner *****************************/
    // Graph(std::string file, int input_type);
    // ~Graph() {}

    // Graph(const Graph &)=delete; // disable copy constructor
    // Graph& operator=(const Graph &)=delete; // disable assignment operator

// protected:
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


    /*********************************** GraphPi *******************************/
public:
    int v_cnt; // number of vertex
    unsigned int e_cnt; // number of edge
    long long tri_cnt; // number of triangle
    double max_running_time = 60 * 60 * 24; // second

    int *edge; // edges
    unsigned int *vertex; // v_i's neighbor is in edge[ vertex[i], vertex[i+1]-1]
    
    Graph() {
        v_cnt = 0;
        e_cnt = 0;
        edge = nullptr;
        vertex = nullptr;
    }

    ~Graph() {
        if(edge != nullptr) delete[] edge;
        if(vertex != nullptr) delete[] vertex;
    }

    int intersection_size(int v1,int v2);
    int intersection_size_clique(int v1,int v2);

    //single thread triangle counting
    long long triangle_counting();
    
    //multi thread triangle counting
    long long triangle_counting_mt(int thread_count);

    //general pattern matching algorithm with multi thread
    long long pattern_matching(const Schedule& schedule, int thread_count, bool clique = false);

    //this function will be defined at code generation
    long long unfold_pattern_matching(const Schedule& schedule, int thread_count, bool clique = false);

    //general pattern matching algorithm with multi thread ans multi process
    long long pattern_matching_mpi(const Schedule& schedule, int thread_count, bool clique = false);

    // int max_degree;

    void tc_mt(long long * global_ans);

    void get_edge_index(int v, unsigned int& l, unsigned int& r) const;

    void pattern_matching_func(const Schedule& schedule, VertexSet* vertex_set, VertexSet& subtraction_set, long long& local_ans, int depth, bool clique = false);

    void pattern_matching_aggressive_func(const Schedule& schedule, VertexSet* vertex_set, VertexSet& subtraction_set, VertexSet& tmp_set, long long& local_ans, int depth);

    //this function will be defined at code generation
    void unfold_pattern_matching_aggressive_func(const Schedule& schedule, VertexSet* vertex_set, VertexSet& subtraction_set, VertexSet& tmp_set, long long& local_ans, int depth);

private:
    friend Graphmpi;
    
};
