// This code is modified from DistGraph:
// https://github.com/zakimjz/DistGraph.git
#pragma once
#include "graph.h"

struct SEdge {
  vidType src;
  vidType dst;
  eidType id;
  SEdge() {}
  SEdge(vidType _src, vidType _dst) : src(_src), dst(_dst), id(0) {}
  SEdge(vidType _src, vidType _dst, eidType _eid) : src(_src), dst(_dst), id(_eid) {}
  std::string to_string() const {
    std::stringstream ss;
    ss << "e(" << src << "," << dst << ")";
    return ss.str();
  }
};

typedef std::vector<vidType> RMPath;

struct LabEdge {
  vidType src;
  vidType dst;
  label_t elabel;
  unsigned eid;
  LabEdge() : src(0), dst(0), elabel(0), eid(0) {}
  LabEdge(vidType src_, vidType dst_, label_t elabel_, unsigned eid_)
      : src(src_), dst(dst_), elabel(elabel_), eid(eid_) {}
  std::string to_string() const {
    std::stringstream ss;
    ss << "e(" << src << "," << dst << "," << elabel << ")";
    return ss.str();
  }
};

typedef std::vector<LabEdge*> LabEdgeList;

// Used for constructing canonical graph
class Vertex {
public:
  label_t label;
  std::vector<SEdge> edges; // neighbor list
  void push(vidType src, vidType dst, label_t) {
    edges.resize(edges.size() + 1);
    edges[edges.size() - 1].src = src;
    edges[edges.size() - 1].dst = dst;
    return;
  }
  std::string to_string() const {
    std::stringstream ss;
    ss << "vlabel: " << label << ", ";
    for (auto e : edges) ss << e.to_string() << " ";
    return ss.str();
  }
};

// Canonical graph used for canonical check.
// A pattern (DFSCode) is converted to a canonical graph
// to perform a canonical check (minimal DFSCode)
class CGraph : public std::vector<Vertex> {
private:
  unsigned num_edges;
  bool directed;

public:
  typedef std::vector<Vertex>::iterator vertex_iterator;
  CGraph() : num_edges(0), directed(false) {}
  CGraph(bool _directed) { directed = _directed; }
  unsigned edge_size() const { return num_edges; }
  unsigned vertex_size() const { return (unsigned)size(); } // wrapper
  bool is_directed() const { return directed; }
  void buildEdge() {
    std::pair<unsigned, unsigned> new_edge;
    std::map<std::pair<unsigned, unsigned>, unsigned> edge_map;
    unsigned id = 0;
    for (vidType src = 0; src < vidType(size()); ++src) {
      for (auto it = (*this)[src].edges.begin(); it != (*this)[src].edges.end(); ++it) {
        auto dst = it->dst;
        if (directed || src <= dst)
          new_edge = std::make_pair(src, dst);
        else
          new_edge = std::make_pair(dst, src);
        // Assign unique id's for the edges.
        if (edge_map.find(new_edge) == edge_map.end()) {
          edge_map[new_edge] = id;
          ++id;
        } else {
          // it->id = edge_map[new_edge];
        }
      }
    }
    num_edges = id;
  }
  std::string to_string() const {
    std::stringstream ss;
    for (size_t vid = 0; vid < size(); ++vid)
      ss << vid << ": " << (*this)[vid].to_string() << "\n";
    return ss.str();
  }
};

// A 5-tuple element in a DFSCode
class DFS {
public:
  vidType from;      // source vertex
  vidType to;        // target vertex
  label_t fromlabel; // source vertex label
  label_t elabel;    // edge label
  label_t tolabel;   // target vertex label
  friend bool operator==(const DFS& d1, const DFS& d2) {
    return (d1.from == d2.from && d1.to == d2.to &&
            d1.fromlabel == d2.fromlabel && d1.elabel == d2.elabel &&
            d1.tolabel == d2.tolabel);
  }
  friend bool operator!=(const DFS& d1, const DFS& d2) { return (!(d1 == d2)); }
  friend std::ostream& operator<<(std::ostream& out, const DFS& d) {
    out << d.to_string().c_str();
    return out;
  }
  friend bool operator<(const DFS& d1, const DFS& d2) {
    if (d1.from < d2.from)
      return true;
    if (d1.from > d2.from)
      return false;
    if (d1.to < d2.to)
      return true;
    if (d1.to > d2.to)
      return false;
    if (d1.fromlabel < d2.fromlabel)
      return true;
    if (d1.fromlabel > d2.fromlabel)
      return false;
    if (d1.elabel < d2.elabel)
      return true;
    if (d1.elabel > d2.elabel)
      return false;
    if (d1.tolabel < d2.tolabel)
      return true;
    if (d1.tolabel > d2.tolabel)
      return false;
    return false;
  }

  DFS() : from(0), to(0), fromlabel(0), elabel(0), tolabel(0) {}
  DFS(vidType from, vidType to, label_t fromlabel, label_t elabel, label_t tolabel)
      : from(from), to(to), fromlabel(fromlabel), elabel(elabel),
        tolabel(tolabel) {}
  DFS(char* buffer, int size);
  DFS(const DFS& d)
      : from(d.from), to(d.to), fromlabel(d.fromlabel), elabel(d.elabel),
        tolabel(d.tolabel) {}
  std::string to_string(bool print_edge_type = true) const {
    std::stringstream ss;
    if (print_edge_type) {
      if (is_forward())
        ss << "F";
      else
        ss << "B";
    }
    ss << "(" << from << " " << to << " " << fromlabel << " " << elabel << " "
       << tolabel << ")";
    return ss.str();
  }
  bool is_forward() const { return from < to; }
  bool is_backward() const { return from > to; }
};

// DFSCode (pattern) is a sequence of 5-tuples
struct DFSCode : public std::vector<DFS> {
private:
  RMPath rmpath; // right-most path
public:
  const RMPath& get_rmpath() const { return rmpath; }
  // RMPath is in the opposite order than the DFS code, i.e., the
  // indexes into DFSCode go from higher numbers to lower numbers.
  const RMPath& buildRMPath() {
    rmpath.clear();
    vidType old_from = (vidType)-1;
    for (int i = size() - 1; i >= 0; --i) {
      if ((*this)[i].from < (*this)[i].to && // forward
          (rmpath.empty() || old_from == (*this)[i].to)) {
        rmpath.push_back(i);
        old_from = (*this)[i].from;
      }
    }
    return rmpath;
  }
  // Convert current DFS code into a canonical graph.
  bool toGraph(CGraph& g) const {
    g.clear();
    for (auto it = begin(); it != end(); ++it) {
      g.resize(std::max(it->from, it->to) + 1);
      if (it->fromlabel != (label_t)-1)
        g[it->from].label = it->fromlabel;
      if (it->tolabel != (label_t)-1)
        g[it->to].label = it->tolabel;
      g[it->from].push(it->from, it->to, it->elabel);
      if (g.is_directed() == false)
        g[it->to].push(it->to, it->from, it->elabel);
    }
    g.buildEdge();
    return (true);
  }
  // Return number of nodes in the graph.
  unsigned nodeCount(void) {
    unsigned nodecount = 0;
    for (DFSCode::iterator it = begin(); it != end(); ++it)
      nodecount =
          std::max(nodecount, (unsigned)(std::max(it->from, it->to) + 1));
    return (nodecount);
  }
  DFSCode& operator=(const DFSCode& other) {
    if (this == &other)
      return *this;
    std::vector<DFS>::operator=(other);
    rmpath                    = other.rmpath;
    return *this;
  }
  friend bool operator==(const DFSCode& d1, const DFSCode& d2) {
    if (d1.size() != d2.size())
      return false;
    for (size_t i = 0; i < d1.size(); i++)
      if (d1[i] != d2[i])
        return false;
    return true;
  }
  friend bool operator<(const DFSCode& d1, const DFSCode& d2) {
    if (d1.size() < d2.size())
      return true;
    else if (d1.size() > d2.size())
      return false;
    for (size_t i = 0; i < d1.size(); i++) {
      if (d1[i] < d2[i])
        return true;
      else if (d2[i] < d1[i])
        return false;
    }
    return false; // equal
  }
  friend std::ostream& operator<<(std::ostream& out, const DFSCode& code);
  void push(vidType from, vidType to, label_t fromlabel, label_t elabel,
            label_t tolabel) {
    resize(size() + 1);
    DFS& d      = (*this)[size() - 1];
    d.from      = from;
    d.to        = to;
    d.fromlabel = fromlabel;
    d.elabel    = elabel;
    d.tolabel   = tolabel;
  }
  void pop() { resize(size() - 1); }
  std::string to_string(bool print_edge_type = true) const {
    if (empty())
      return "";
    std::stringstream ss;
    size_t i = 0;
    ss << (*this)[i].to_string(print_edge_type);
    i++;
    for (; i < size(); ++i) {
      ss << ";" << (*this)[i].to_string(print_edge_type);
    }
    return ss.str();
  }
};

std::ostream& operator<<(std::ostream& out, const DFSCode& code) {
  out << code.to_string();
  return out;
}

// An embedding consists of an edge (pointer)
// and an embedding pointer to its parent embedding
struct BaseEdgeEmbedding {
  unsigned num_vertices;
  eidType edge; // edge id
  SEdge* sedge; // edge pointer
  BaseEdgeEmbedding* prev;
  BaseEdgeEmbedding() :
    BaseEdgeEmbedding(0, 0, NULL, NULL) {}
  BaseEdgeEmbedding(vidType nv, eidType eid, BaseEdgeEmbedding* p) : 
    BaseEdgeEmbedding(nv, eid, NULL, p) {}
  BaseEdgeEmbedding(vidType nv, SEdge* se, BaseEdgeEmbedding* p) : 
    BaseEdgeEmbedding(nv, 0, se, p) {}
  BaseEdgeEmbedding(vidType nv, eidType eid, SEdge* se, BaseEdgeEmbedding* p) : 
    num_vertices(nv), edge(eid), sedge(se), prev(p) {}
  void init(vidType nv, eidType eid, SEdge* se, BaseEdgeEmbedding* p) {
    num_vertices = nv;
    edge = eid;
    sedge = se;
    prev = p;
  }
  std::string to_string() const {
    std::stringstream ss;
    if (sedge) ss << "[" << sedge->to_string() << "]";
    else ss << "[" << edge << "]";
    return ss.str();
  }
};

struct LabEdgeEmbedding {
  unsigned num_vertices;
  LabEdge* edge;
  LabEdgeEmbedding* prev;
  LabEdgeEmbedding() : num_vertices(0), edge(0), prev(0){};
  std::string to_string() const {
    std::stringstream ss;
    ss << "[" << edge->to_string() << "]";
    return ss.str();
  }
  std::string to_string_all() {
    std::vector<LabEdge> ev;
    ev.push_back(*edge);
    for (LabEdgeEmbedding* p = prev; p; p = p->prev) {
      ev.push_back(*(p->edge));
    }
    std::reverse(ev.begin(), ev.end());
    std::stringstream ss;
    for (size_t i = 0; i < ev.size(); i++) {
      ss << ev[i].to_string() << "; ";
    }
    return ss.str();
  }
};

// Embedding list
class BaseEdgeEmbeddingList : public std::vector<BaseEdgeEmbedding> {
public:
  void push(int n, SEdge* edge, BaseEdgeEmbedding* prev) {
    BaseEdgeEmbedding emb;
    emb.init(n, 0, edge, prev);
    push_back(emb);
  }
  void push(int n, eidType edge, BaseEdgeEmbedding* prev) {
    BaseEdgeEmbedding emb;
    emb.init(n, edge, NULL, prev);
    push_back(emb);
  }
  std::string to_string() const {
    std::stringstream ss;
    for (size_t i = 0; i < size(); i++)
      ss << (*this)[i].to_string() << "; ";
    return ss.str();
  }
};

class LabEdgeEmbeddingList : public std::vector<LabEdgeEmbedding> {
public:
  void push(int n, LabEdge* edge, LabEdgeEmbedding* prev) {
    LabEdgeEmbedding d;
    d.num_vertices = n;
    d.edge         = edge;
    d.prev         = prev;
    push_back(d);
  }
  std::string to_string() const {
    std::stringstream ss;
    for (size_t i = 0; i < size(); i++)
      ss << (*this)[i].to_string() << "; ";
    return ss.str();
  }
};

typedef std::map<int, std::map<int, BaseEdgeEmbeddingList>> EmbeddingLists2D;
typedef std::map<int, BaseEdgeEmbeddingList> EmbeddingLists1D;
typedef std::map<int, std::map<int, std::map<int, LabEdgeEmbeddingList>>>
    LabEmbeddingLists3D;
typedef std::map<int, std::map<int, LabEdgeEmbeddingList>> LabEmbeddingLists2D;
typedef std::map<int, LabEdgeEmbeddingList> LabEmbeddingLists1D;

// Stores information of edges/nodes that were already visited in the
// current DFS branch of the search.
class History : public std::vector<eidType> {
private:
  std::set<vidType> vertex;

public:
  History() {}
  History(Graph &g, BaseEdgeEmbedding* p) { build(g, p); }
  bool hasVertex(unsigned id) { return (bool)vertex.count(id); }
  bool hasEdge(Graph &g, vidType u, vidType v) {
    for (auto e = begin(); e < end(); e++) {
      auto src = g.get_src(*e);
      auto dst = g.get_dst(*e);
      if (src == u && dst == v) return true;
      else if (src == v && dst == u) return true;
    }
    return false;
  }
  void build(Graph &g, BaseEdgeEmbedding* emb) {
    if (emb) {
      auto u = g.get_src(emb->edge);
      auto v = g.get_dst(emb->edge);
      push_back(emb->edge);
      vertex.insert(u);
      vertex.insert(v);
      for (BaseEdgeEmbedding* p = emb->prev; p; p = p->prev) {
        push_back(p->edge); // this line eats 8% of overall instructions(!)
        auto src = g.get_src(p->edge);
        auto dst = g.get_dst(p->edge);
        vertex.insert(src);
        vertex.insert(dst);
      }
      std::reverse(begin(), end());
    }
  }
  std::string to_string() const {
    std::stringstream ss;
    for (size_t i = 0; i < size(); i++) {
      ss << at(i) << "; ";
    }
    return ss.str();
  }
};

class LabHistory : public std::vector<LabEdge*> {
private:
  std::set<int> edge;
  std::set<int> vertex;

public:
  bool hasEdge(unsigned id) { return (bool)edge.count(id); }
  bool hasLabEdge(LabEdge e) {
    for (auto it = this->begin(); it != this->end(); ++it) {
      // if((*it)->src == e.src && (*it)->dst == e.dst && (*it)->elabel ==
      // e.elabel)
      if ((*it)->src == e.src && (*it)->dst == e.dst)
        return true;
      // else if((*it)->src == e.dst && (*it)->dst == e.src && (*it)->elabel ==
      // e.elabel)
      else if ((*it)->src == e.dst && (*it)->dst == e.src)
        return true;
    }
    return false;
  }
  bool hasVertex(unsigned id) { return (bool)vertex.count(id); }
  LabHistory() {}
  LabHistory(LabEdgeEmbedding* p) { build(p); }
  void build(LabEdgeEmbedding* e) {
    if (e) {
      push_back(e->edge);
      edge.insert(e->edge->eid);
      vertex.insert(e->edge->src);
      vertex.insert(e->edge->dst);
      for (LabEdgeEmbedding* p = e->prev; p; p = p->prev) {
        push_back(p->edge); // this line eats 8% of overall instructions(!)
        edge.insert(p->edge->eid);
        vertex.insert(p->edge->src);
        vertex.insert(p->edge->dst);
      }
      std::reverse(begin(), end());
    }
  }
  std::string to_string() const {
    std::stringstream ss;
    for (size_t i = 0; i < size(); i++) {
      ss << at(i)->to_string() << "; ";
    }
    return ss.str();
  }
};

