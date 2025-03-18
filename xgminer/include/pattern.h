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
#include "common.h"
#include "utils.h"

#ifndef INDEX
#define INDEX(x,y,n) ((x)*(n)+(y)) 
#endif

enum PatternType {
    Rectangle,
    QG3, // quasi-clique of size 4
    Pentagon,
    House,
    Hourglass,
    Cycle_6_Tri,
    Clique_7_Minus
};

// GLUMIN
enum Labelling {
    UNLABELLED,
    LABELLED,
    PARTIALLY_LABELLED,
    DISCOVER_LABELS
};



class Pattern {
/*********************************** GLUMIN *******************************/
#if 1
public:
    Pattern() : Pattern("", false) { }
    Pattern(std::string name) : Pattern(name, 0, 0) { }
    Pattern(std::string name, int nv, int ne) : name_(name), num_vertices(nv), num_edges(ne) { }
    Pattern(std::string filename, bool is_labeled) : 
        name_(""), has_label(is_labeled), core_length_(0) {
        read_adj_file(filename);
        if (has_label) labelling = LABELLED;
        set_name();
    }
    // ~Pattern() {}
    bool is_clique() const { return num_vertices>1 && num_edges == (num_vertices-1)*num_vertices/2; }
    bool is_path() const { return num_edges == num_vertices-1; }
    bool is_chain() const { return num_edges == num_vertices-1; }
    bool is_wedge() const { return name_ == "wedge"; }
    bool is_triangle() const { return name_ == "triangle"; }
    // 4-motif
    bool is_diamond() const { return name_ == "diamond"; }
    bool is_rectangle() const { return name_ == "rectangle"; }
    bool is_tailedtriangle() const { return name_ == "tailedtriangle"; }
    bool is_4path() const { return name_ == "4path"; }
    bool is_3star() const { return name_ == "3star"; }
    // 5-motif
    bool is_5path() const { return name_ == "5path"; }
    bool is_pentagon() const { return name_ == "pentagon"; }
    bool is_house() const { return name_ == "house"; }
    bool is_semihouse() const { return name_ == "semihouse"; }
    bool is_closedhouse() const { return name_ == "closedhouse"; }
    bool is_hourglass() const { return name_ == "hourglass"; }
    bool is_taileddiamond() const { return name_ == "taileddiamond"; }
    bool is_taileddiamond2() const { return name_ == "taileddiamond2"; }
    // 6-motif
    bool is_6path() const { return name_ == "6path"; }
    bool is_dumbbell() const { return name_ == "dummbell"; }

    // colorful patterns
    bool is_4color_square() const { return name_ == "4color-square"; }

    bool is_connected(vidType u, vidType v) const;
    void read_adj_file(std::string inputfile);
    std::string get_name() const { return name_; }
    int sizeVerts() const { return num_vertices; }
    int sizeEdges() const { return num_edges; }
    int get_num_vertices() const { return num_vertices; }
    int get_num_edges() const { return num_edges; }
    int get_degree(vidType v) const { return adj_list.at(v).size(); }
    int get_max_degree() const { return max_degree; }

    vidType get_neighbor(vidType v, vidType i) const { return adj_list.at(v)[i]; }
    const std::vector<vidType> &get_neighbours(vidType v) const { return adj_list.at(v); }
    const VertexList& N(vidType v) const { return adj_list.at(v); } 
    VertexList v_list() const;
    std::string to_string() const;
    void print_meta_data() const;

    void set_name();
    Pattern &add_edge_glumin(vidType u, vidType v);
    Pattern &set_label(vidType u, vlabel_t l);
    Pattern &remove_edge(vidType u, vidType v);
    std::string to_string(const std::vector<vlabel_t> &given_labels) const;
    void buildCoreTable();
    void computeKCore();
    friend std::ostream &operator <<(std::ostream &os, const Pattern &p) {
        os << p.to_string();
        return os;
    }
#endif
#if 0
    vlabel_t get_vlabel(vidType v) const { return vlabels[v]; } // vid starting from 0
    elabel_t get_elabel(eidType e) const { return elabels[e]; }
    int get_max_label_frequency() const { return max_label_frequency_; }
    vidType getLabelsFrequency(const vlabel_t label) const { return labels_frequency_.at(label); }
    
    int getCoreValue(const vidType vid) const { return core_table[vid]; }
    // const std::unordered_map<vlabel_t, vidType>* getVertexNLF(const vidType id) const { return nlf_ + id; }
    // Labelling get_labelling() const { return labelling; }
    int get2CoreSize() const { return core_length_; }
    int get_num_labels() { return num_vertex_classes; }
    
    // void set_labelling(Labelling l) { labelling = l; }
    // void BuildNLF();
    // void computeLabelsFrequency();

    bool operator==(const Pattern &p) const { return p.vlabels == vlabels; }
    bool operator<(const Pattern &p) const { return p.vlabels < vlabels; }
#endif

private:
    std::string name_;
    bool has_label;
    std::map<vidType, VertexList> adj_list;
    std::vector<vlabel_t> vlabels;
    std::vector<elabel_t> elabels;
    Labelling labelling = UNLABELLED;
    int num_vertices;
    int num_edges;
    int num_vlabels;
    int num_elabels;
    int max_degree;
    int core_length_;
    int max_label;
    int num_vertex_classes;
    int max_label_frequency_;
    std::vector<int> core_table;
    // std::vector<vidType> labels_frequency_;
    // std::unordered_map<vlabel_t, vidType>* nlf_;


/*********************************** GraphPi *******************************/
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
    
    void PatternType_printer(PatternType type);
    bool is_equal_adj_mat(const int* adj_mat1, const int* adj_mat2, int size);
private:
    int* adj_mat;
    int size;
};

