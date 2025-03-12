#ifndef _PATTERN_H_
#define _PATTERN_H_

#include <iostream>
#include <vector>
#include <algorithm>

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
    ~Pattern() {}
    // bool is_clique() const { return num_vertices>1 && num_edges == (num_vertices-1)*num_vertices/2; }
    // bool is_path() const { return num_edges == num_vertices-1; }
    // bool is_chain() const { return num_edges == num_vertices-1; }
    // bool is_wedge() const { return name_ == "wedge"; }
    // bool is_triangle() const { return name_ == "triangle"; }
};

#endif