/*
    * common.h
    *
    * borrowed from GraphPi and GLUMIN
*/

#pragma once
#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <set>
#include <map>
#include <deque>
#include <vector>
#include <limits>
#include <cstdio>
#include <string>
#include <cstring>
#include <fstream>
#include <sstream>
#include <climits>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <regex>
#include <iterator>
#include <stdexcept>
#include <algorithm>
#include <unordered_map>
#include <sys/time.h>
#include <cstdlib>
#include <string>


/*********************************** GLUMIN *******************************/
typedef float   feat_t;    // vertex feature type
typedef uint8_t patid_t;   // pattern id type
typedef uint8_t mask_t;    // mask type
typedef uint8_t label_t;   // label type
typedef uint8_t vlabel_t;  // vertex label type
typedef uint16_t elabel_t; // edge label type
typedef uint8_t cmap_vt;   // cmap value type
typedef int32_t vidType;   // vertex ID type
typedef int64_t eidType;   // edge ID type
typedef int32_t IndexT;
typedef uint64_t emb_index_t; // embedding index type
typedef unsigned long long AccType;
typedef uint8_t arrayType; // roaring bitmap array ID type
typedef uint8_t typeType; // construct a small bitmap to check chunks type

typedef std::vector<patid_t> PidList;    // pattern ID list
typedef std::vector<vidType> VertexList; // vertex ID list
typedef std::vector<std::vector<vidType>> VertexLists;
typedef std::unordered_map<vlabel_t, int> nlf_map;

#define ADJ_SIZE_THREASHOLD 1024
#define FULL_MASK 0xffffffff
#define MAX_PATTERN_SIZE 8
#define MAX_FSM_PATTERN_SIZE 5
#define NUM_BUCKETS 128
#define BUCKET_SIZE 1024
#define DIVIDE_INTO(x,y) ((x + y - 1)/y)
#define BLOCK_SIZE    256//128//
#define BLOCK_SIZE_DENSE 128
#define WARP_SIZE     32
#define LOG_WARP_SIZE 5
#define DIVIDE_INTO(x,y) ((x + y - 1)/y)
#define MAX_THREADS (30 * 1024)
#define WARPS_PER_BLOCK (BLOCK_SIZE / WARP_SIZE)
#define WARPS_PER_BLOCK_DENSE (BLOCK_SIZE_DENSE / WARP_SIZE)
#define MAX_BLOCKS (MAX_THREADS / BLOCK_SIZE)
#define NUM_WARPS (BLOCK_SIZE / WARP_SIZE)
#define BYTESTOMB(memory_cost) ((memory_cost)/(double)(1024 * 1024))

#define THREAD_GROUP_SIZE 32 //Set 8 or 16 or 32
#define THREAD_GROUP_PER_BLOCK (BLOCK_SIZE / THREAD_GROUP_SIZE)
#define THREAD_GROUP_PER_WARP (WARP_SIZE / THREAD_GROUP_SIZE)

// typedef uint32_t bitmapType;
#define bitmapType vidType
#define BITMAP_WIDTH 32
#define SUBTASK_SIZE 32

// roaring bitmap
// #define ROARING
#define CHUNK_WIDTH 256
#define ARRAY_ID_WIDTH 8
#define ARRAY_LIMIT 32
#define PAD_PER_CHUNK (CHUNK_WIDTH / BITMAP_WIDTH)
#define TYPE_WIDTH 8

// #define LOAD_SRAM
#define LOAD_RATE 8
#define CUDA_SELECT_DEVICE 4

// worklist
#define WORKLIST_SIZE 0x48000
#define WORKLIST_THRESHOLD WORKLIST_SIZE * 0.6
#define LOCAL_SHORT_WORKLIST_SIZE (WARP_SIZE * 2)
#define LOCAL_SHORT_WORKLIST_THRESHOLD (LOCAL_SHORT_WORKLIST_SIZE * 0.8)
// #define LOCAL_BITMAP_LIST_SIZE (LOCAL_SHORT_WORKLIST_SIZE * 0.05)
#define LOCAL_BITMAP_LIST_SIZE 32

#define SPARSE_THRESHOLD 4
#define SMALL_SPLIT 1
#define SPARSE_SPLIT 2
#define DENSE_SPLIT 3

#define WARP_LIMIT 0
#define BLOCK_LIMIT 4096
#define BLOCK_GROUP 80

enum Status {
  Idle,
  Extending,
  IteratingEdge,
  Working,
  ReWorking
};

#define OP_INTERSECT 'i'
#define OP_DIFFERENCE 'd'
extern std::map<char,double> time_ops;

const std::string long_separator = "--------------------------------------------------------------------\n";
const std::string short_separator = "-----------------------\n";


#include <cuda_runtime.h>
#include <cuda_runtime_api.h>




/*********************************** GraphPi *******************************/


double get_wall_time(); 

// void PatternType_printer(PatternType type);

// bool is_equal_adj_mat(const int* adj_mat1, const int* adj_mat2, int size);

// void GetDataType(DataType &type, const std::string str);

int read_int();

unsigned int read_unsigned_int();


typedef int32_t v_index_t;
typedef int64_t e_index_t;



/*********************************** X-GMiner *******************************/

#define BITMAP64_WIDTH 64
#define bitmap64_Type uint64_t



enum Input_FileFormat {
    SNAP_TXT = 0,
    BINARY = 1,
    DIMACS = 2,
    MatrixMarket = 3,
    INVALID = 4,
};


bool checkFileExtension(std::string filename, const std::string& extension);

bool checkDirectoryForFilesWithExtension(std::string directoryPath, const std::string& extension);

Input_FileFormat getFileFormat(std::string filename);


enum XGMinerPatternType {
    TRIANGLE = 1,
    RECTANGLE = 2,
    DIAMOND = 3,

    P1_GraphPi = 4, // HOUSE
    P2_GraphPi_Paper = 5, // Cycle_6_Tri
    P3_GraphPi_Paper = 6,
    P4_GraphPi = 7,
    P5_GraphPi = 8,
    P6_GraphPi = 9,

    QuasiG3 = 10, // quasi-clique of size 4
    PENGTAGON = 11, // Pentagon  supported by GraphPi
    HOUSE = 13, // House  supported by GraphPi
    HOURGLASS = 14, // Hourglass  supported by GraphPi
    CYCLE_6_TRI = 15, // Cycle_6_Tri  supported by GraphPi
    CLIQUE_7_MINUS = 16, // Clique_7_Minus  supported by GraphPi

    P1_GLUMIN = 17,
    P2_GLUMIN = 18,
    P3_GLUMIN = 19,
};

static int global_pattern_ID_map_to_graphpi[] = {
    -1,0,-1,-1,-1,-1,-1,-1,-1,1,2,3,4,5,6
};

// static int global_pattern_ID_map_to_glumin[] = {
//     -1,1,2,3,4,5,6,7,8,9,10,11,13,14,15,16
// };