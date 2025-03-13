/*
    * pattern.h
    *
    * borrowed from GraphPi and GLUMIN
*/

#pragma once
#include <iostream>
#include <vector>
#include <algorithm>
#include <set>
#include <vector>

#ifndef INDEX
#define INDEX(x,y,n) ((x)*(n)+(y)) 
#endif

enum PatternType {
    Rectangle,
    QG3,
    Pentagon,
    House,
    Hourglass,
    Cycle_6_Tri,
    Clique_7_Minus
};


class Pattern {
public:
    Pattern();// : Pattern("", false) { }
    // Pattern(std::string name) : Pattern(name, 0, 0) { }
    // Pattern(std::string name, int nv, int ne) : name_(name), num_vertices(nv), num_edges(ne) { }
    // Pattern(std::string filename, bool is_labeled) : 
    //     name_(""), has_label(is_labeled), core_length_(0) {
    //     read_adj_file(filename);
    //     if (has_label) labelling = LABELLED;
    //     set_name();
    // }
    // ~Pattern() {}
    // bool is_clique() const { return num_vertices>1 && num_edges == (num_vertices-1)*num_vertices/2; }
    // bool is_path() const { return num_edges == num_vertices-1; }
    // bool is_chain() const { return num_edges == num_vertices-1; }
    // bool is_wedge() const { return name_ == "wedge"; }
    // bool is_triangle() const { return name_ == "triangle"; }


public:
    Pattern(int _size, bool clique = false);
    Pattern(int _size, char* buffer);
    ~Pattern();
    Pattern(const Pattern& p);
    Pattern(PatternType type);
    void add_edge(int x, int y);
    void del_edge(int x, int y);
    inline void add_ordered_edge(int x, int y) { adj_mat[INDEX(x, y, size)] = 1;}
    inline int get_size() const {return size;}
    inline const int* get_adj_mat_ptr() const {return adj_mat;}
    bool check_connected() const;
    void count_all_isomorphism(std::set< std::set<int> >& s) const;
    void print() const;
    bool is_dag() const;
    Pattern& operator =(const Pattern&);
    void get_full_permutation(std::vector< std::vector<int> >& vec, bool use[], std::vector<int> tmp_vec, int depth) const;
    
private:
    int* adj_mat;
    int size;
};

