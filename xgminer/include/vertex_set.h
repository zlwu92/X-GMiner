/*
    * vertex_set.h
    *
    * borrowed from GraphPi and GLUMIN
*/

#pragma once
#include "schedule.h"

#ifdef USE_NUMA
  #include <numa.h>
  template<typename T>
  T* custom_alloc_local(size_t elements) {
    return (T*)numa_alloc_local(sizeof(T) * elements);
    //return (T*)aligned_alloc(PAGE_SIZE, elements * sizeof(T));
  }
  template<typename T>
  T* custom_alloc_global(size_t elements) {
    return (T*)numa_alloc_interleaved(sizeof(T) * elements);
  }
  template<typename T>
  void custom_free(T *ptr, size_t elements) {
    numa_free(ptr, sizeof(T)*elements);
  }
#else
  template<typename T>
  T* custom_alloc_local(size_t elements) {
    return new T[elements];
  }
  template<typename T>
  T* custom_alloc_global(size_t elements) {
    return new T[elements];
  }
  template<typename T>
  void custom_free(T *ptr, size_t elements) {
    delete[] ptr;
  }
#endif

template<typename T>
static void read_file(std::string fname, T *& pointer, size_t length) {
  pointer = custom_alloc_global<T>(length);
  assert(pointer);
  std::ifstream inf(fname.c_str(), std::ios::binary);
  if (!inf.good()) {
    std::cerr << "Failed to open file: " << fname << "\n";
    exit(1);
  }
  inf.read(reinterpret_cast<char*>(pointer), sizeof(T) * length);
  inf.close();
}

template<typename T>
static void map_file(std::string fname, T *& pointer, size_t length) {
  int inf = open(fname.c_str(), O_RDONLY, 0);
  if (-1 == inf) {
    std::cerr << "Failed to open file: " << fname << "\n";
    exit(1);
  }
  pointer = (T*)mmap(nullptr, sizeof(T) * length, PROT_READ, MAP_SHARED, inf, 0);
  assert(pointer != MAP_FAILED);
  close(inf);
}

constexpr vidType VID_MIN = 0;
constexpr vidType VID_MAX = std::numeric_limits<vidType>::max();

inline vidType bs(vidType* ptr, vidType set_size, vidType o){
  vidType idx_l = -1;
  vidType idx_r = set_size;
  //guarantees in this area is that idx_l is before where we would put o 
  while(idx_r-idx_l > 1){
    vidType idx_t = (idx_l+idx_r)/2;
    if(ptr[idx_t] < o)idx_l = idx_t;
    else idx_r = idx_t;
  }
  return idx_l+1;
}

class VertexSet {

/*********************************** GLUMIN *******************************/
private: // memory managed regions for per-thread intermediates
    static thread_local std::vector<vidType*> buffers_exist, buffers_avail;
public:
    static void release_buffers();
    static vidType MAX_DEGREE;
    vidType *ptr;
private:
    vidType set_size, vid;
    const bool pooled;

public:
    VertexSet() : set_size(0), vid(-1), pooled(true), data(nullptr), size(0), allocate(false){
        if(buffers_avail.size() == 0) { 
            vidType *p = custom_alloc_local<vidType>(MAX_DEGREE);
            buffers_exist.push_back(p);
            buffers_avail.push_back(p);
        }
        ptr = buffers_avail.back();
        buffers_avail.pop_back();
    }
    VertexSet(vidType *p, vidType s, vidType id) : 
        ptr(p), set_size(s), vid(id), pooled(false) {}
    
    VertexSet(const VertexSet&)=delete;
    VertexSet& operator=(const VertexSet&)=delete;
    VertexSet(VertexSet&&)=default;
    VertexSet& operator=(VertexSet&&)=default;
    
    // ~VertexSet() {
    //     if(pooled) {
    //     buffers_avail.push_back(ptr);
    //     }
    // }
#if 1
    vidType getsize() const { return set_size; }
    VertexSet operator &(const VertexSet &other) const {
        VertexSet out;
        vidType idx_l = 0, idx_r = 0;
        while(idx_l < set_size && idx_r < other.set_size) {
        vidType left = ptr[idx_l];
        vidType right = other.ptr[idx_r];
        if(left <= right) idx_l++;
        if(right <= left) idx_r++;
        if(left == right) out.ptr[out.set_size++] = left;
        }
        return out;
    }
    uint32_t get_intersect_num(const VertexSet &other) const {
        uint32_t num = 0;
        vidType idx_l = 0, idx_r = 0;
        while(idx_l < set_size && idx_r < other.set_size) {
        vidType left = ptr[idx_l];
        vidType right = other.ptr[idx_r];
        if(left <= right) idx_l++;
        if(right <= left) idx_r++;
        if(left == right) num++;
        }
        return num;
    }

    void print() const {
        std::copy(ptr, ptr+set_size, std::ostream_iterator<vidType>(std::cout, " "));
    }

    vidType difference_buf(vidType *outBuf, const VertexSet &other) const;

    VertexSet operator -(const VertexSet &other) const {
        VertexSet out;
        out.set_size = difference_buf(out.ptr, other); 
        return out;
    }

    VertexSet& difference(VertexSet& dst, const VertexSet &other) const {
        dst.set_size = difference_buf(dst.ptr, other);
        return dst;
    }

    VertexSet intersect(const VertexSet &other, vidType upper) const {
        VertexSet out;
        vidType idx_l = 0, idx_r = 0;
        while(idx_l < set_size && idx_r < other.set_size) {
        vidType left = ptr[idx_l];
        vidType right = other.ptr[idx_r];
        if(left >= upper) break;
        if(right >= upper) break;
        if(left <= right) idx_l++;
        if(right <= left) idx_r++;
        if(left == right) out.ptr[out.set_size++] = left;
        }
        return out;
    }

    vidType intersect_ns(const VertexSet &other, vidType upper) const {
        vidType idx_l = 0, idx_r = 0, idx_out = 0;
        while(idx_l < set_size && idx_r < other.set_size) {
        vidType left = ptr[idx_l];
        vidType right = other.ptr[idx_r];
        if(left >= upper) break;
        if(right >= upper) break;
        if(left <= right) idx_l++;
        if(right <= left) idx_r++;
        if(left == right) idx_out++;
        }
        return idx_out;
    }

    vidType intersect_ns_except(const VertexSet &other, vidType upper, vidType ancestor) const {
        vidType idx_l = 0, idx_r = 0, idx_out = 0;
        while(idx_l < set_size && idx_r < other.set_size) {
        vidType left = ptr[idx_l];
        vidType right = other.ptr[idx_r];
        if(left >= upper) break;
        if(right >= upper) break;
        if(left <= right) idx_l++;
        if(right <= left) idx_r++;
        if(left == right && left != ancestor) idx_out++;
        }
        return idx_out;
    }

    vidType intersect_except(const VertexSet &other, vidType ancestor) const {
        vidType idx_l = 0, idx_r = 0, idx_out = 0;
        while(idx_l < set_size && idx_r < other.set_size) {
        vidType left = ptr[idx_l];
        vidType right = other.ptr[idx_r];
        if(left <= right) idx_l++;
        if(right <= left) idx_r++;
        if(left == right && left != ancestor) idx_out++;
        }
        return idx_out;
    }

    vidType intersect_except(const VertexSet &other, vidType ancestorA, vidType ancestorB) const {
        vidType idx_l = 0, idx_r = 0, idx_out = 0;
        while(idx_l < set_size && idx_r < other.set_size) {
        vidType left = ptr[idx_l];
        vidType right = other.ptr[idx_r];
        if(left <= right) idx_l++;
        if(right <= left) idx_r++;
        if(left == right && left != ancestorA && left != ancestorB) idx_out++;
        }
        return idx_out;
    }
#endif
    //outBuf may be the same as this->ptr
    vidType difference_buf(vidType *outBuf, const VertexSet &other, vidType upper) const;

    VertexSet difference(const VertexSet &other, vidType upper) const {
        VertexSet out;
        out.set_size = difference_buf(out.ptr, other, upper);
        return out;
    }

    VertexSet& difference(VertexSet& dst, const VertexSet &other, vidType upper) const {
        dst.set_size = difference_buf(dst.ptr, other, upper);
        return dst;
    }

    vidType difference_ns(const VertexSet &other, vidType upper) const;

    VertexSet bounded(vidType up) const {
        if(set_size > 64) {
            vidType idx_l = -1;
            vidType idx_r = set_size;
            while(idx_r-idx_l > 1) {
                vidType idx_t = (idx_l+idx_r)/2;
                if(ptr[idx_t] < up) idx_l = idx_t;
                else idx_r = idx_t;
            }
            return VertexSet(ptr,idx_l+1,vid);
        } else {
            vidType idx_l = 0;
            while(idx_l < set_size && ptr[idx_l] < up) ++idx_l;
            return VertexSet(ptr,idx_l,vid);
        }
    }
    vidType *dataptr() const { return ptr; }
    const vidType *begin() const { return ptr; }
    const vidType *end() const { return ptr+set_size; }
    void add(vidType v) { ptr[set_size++] = v; }
    void clear() { set_size = 0; }
    vidType& operator[](size_t i) { return ptr[i]; }
    const vidType& operator[](size_t i) const { return ptr[i]; }



/*********************************** GraphPi *******************************/
public:
    // VertexSet();
    // allocate new memory according to max_intersection_size
    void init();
    void init(int init_size);
    // use memory from Graph, do not allocate new memory
    void init(int input_size, int* input_data);
    void copy(int input_size, const int* input_data);
    ~VertexSet();
    void intersection(const VertexSet& set0, const VertexSet& set1, int min_vertex = -1, bool clique = false);
    void intersection_with(const VertexSet& set1);
    //set1 is unordered
    static int unorderd_subtraction_size(const VertexSet& set0, const VertexSet& set1, int size_after_restrict = -1);
    void insert_ans_sort(int val);
    inline int get_size() const { return size;}
    inline int get_data(int i) const { return data[i];}
    inline const int* get_data_ptr() const { return data;}
    inline int* get_data_ptr() { return data;}
    inline void push_back(int val) { data[size++] = val;}
    inline void pop_back() { --size;}
    inline int get_last() const { return data[size - 1];}
    bool has_data(int val);
    static int max_intersection_size;
    void build_vertex_set(const Schedule& schedule, const VertexSet* vertex_set, int* input_data, int input_size, int prefix_id, int min_vertex = -1, bool clique = false);


private:
    int* data;
    int size;
    bool allocate;
};
