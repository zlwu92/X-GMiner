#pragma once
#include <iostream>
#include <assert.h>
#include "utils.h"
#include "../include/graph.h"
#include "../include/dataloader.h"
#include "../include/pattern.h"
#include "../include/schedule.h"
#include "../include/common.h"
#include "cmd_option.h"
#include "kernel.h"

class CPU_Baseline {
public:
    CPU_Baseline(Command_Option &opts) {
        data_name = opts.data_name;
        data_path = opts.datagraph_file;
        use_graphpi_sched = opts.use_graphpi_sched;
        run_graphpi = opts.run_graphpi;
        run_our_baseline = opts.run_our_baseline;
        pattern_size = opts.pattern_size;
        adj_mat = opts.pattern_adj_mat;
        patternID = opts.patternID;
        file_format = getFileFormat(data_path);
        do_validation = opts.do_validation;
        vert_induced = opts.vert_induced;
        std::cout << "vertex induced: " << vert_induced << std::endl;
    }

    ~CPU_Baseline() {
        std::cout << "CPU_Baseline destructor" << std::endl;
    }

    void run_baseline_with_graphpi_sched();

    long long run_graphpi_test();

    void run_our_baseline_test();

    void run() {
        local_patternId = patternID - 16;
        if (run_graphpi) {
            LOG_INFO("Running GraphPi algorithm.");
            run_graphpi_test();
        }
        if (run_our_baseline) {
            LOG_INFO("Running our baseline implementation.");
            load_graph_data_from_file();
            if (use_graphpi_sched) {
                run_baseline_with_graphpi_sched();
            } else {
                run_our_baseline_test();
            }
        }
    }

    void load_graph_data_from_file() {
        if (file_format == Input_FileFormat::SNAP_TXT) {
            std::ifstream file(data_path);
            file >> vertices >> edges;
            std::cout << "Vertices: " << vertices << ", Edges: " << edges << std::endl;
            edgeLists.resize(vertices);
            std::map<int,int> id_map;
            int source, target;
            int real_vnum = 0, real_enum = 0;
            while (file >> source >> target) {
                edgeLists[source].insert(target);
                edgeLists[target].insert(source);
                // printf("source=%d target=%d\n", source, target);

                // if(!id_map.count(source))    id_map[source] = real_vnum++;
                // if(!id_map.count(target))    id_map[target] = real_vnum++;
                // source = id_map[source];
                // target = id_map[target];
                
                // edgeLists[source].insert(target);
                // edgeLists[target].insert(source);
                real_enum += 2;
            }

            if (real_vnum != vertices || real_enum != edges * 2) {
                std::cout << "real_vnum=" << real_vnum << " real_enum=" << real_enum << std::endl;
                // delete g;
                file.close();
                LOG_ERROR("Invalid vertex num / edge num.");
            }

            // for (int i = 0; i < vertices; i++) {
            //     std::cout << "Vertex " << i << " edges: ";
            //     for (int edge : edgeLists[i]) {
            //         std::cout << edge << " ";
            //     }
            //     std::cout << std::endl;
            // }

            file.close();
        } else if (file_format == Input_FileFormat::BINARY) {
            std::string inputfile_path;
            std::string prefix = data_path + "/graph";
            std::string name_;
            size_t i = prefix.rfind('/', prefix.length());
            if (i != std::string::npos) inputfile_path = prefix.substr(0, i);
            i = inputfile_path.rfind('/', inputfile_path.length());
            if (i != std::string::npos) name_ = inputfile_path.substr(i+1);
            std::cout << "input file path: " << inputfile_path << ", graph name: " << name_ << "\n";

            std::cout << prefix + ".meta.txt" << "\n";
            std::ifstream f_meta((prefix + ".meta.txt").c_str());
            assert(f_meta);
            int vid_size = 0, eid_size = 0, vlabel_size = 0, elabel_size = 0;
            eidType n_edges;
            vidType n_vertices;
            vidType max_degree;
            int feat_len;
            int num_vertex_classes;
            int num_edge_classes;
            f_meta >> n_vertices;
            f_meta >> n_edges >> vid_size >> eid_size >> vlabel_size >> elabel_size
                    >> max_degree >> feat_len >> num_vertex_classes >> num_edge_classes;
            LOG_INFO("Graph meta information: |V|: " + std::to_string(n_vertices) + 
                    ", |E|: " + std::to_string(n_edges) + ", Max Degree: " + std::to_string(max_degree));
            LOG_INFO("vid_size: " + std::to_string(vid_size) + ", eid_size: " + std::to_string(eid_size) + 
                    ", vlabel_size: " + std::to_string(vlabel_size) + ", elabel_size: " + std::to_string(elabel_size));
            LOG_INFO("Vertex-|\u03A3|: " + std::to_string(num_vertex_classes) + ", Edge-|\u03A3|: " + std::to_string(num_edge_classes));

            vidType *edges1;
            eidType *vertices1;

            // read row pointers
            read_file(prefix + ".vertex.bin", vertices1, n_vertices+1);
            // read column indices
            read_file(prefix + ".edge.bin", edges1, n_edges);

            vertices = n_vertices;
            edges = n_edges;
            edgeLists.resize(vertices);
            for (int i = 0; i < vertices; i++) {
                for (int j = vertices1[i]; j < vertices1[i+1]; j++) {
                    edgeLists[i].insert(edges1[j]);
                }
            }

        } else {
            LOG_ERROR("Invalid file format.");
        }
    }

    void validation() {
        long long ans = run_graphpi_test();
        if (total_count == ans) {
            LOG_INFO("Validation passed.");
        } else {
            LOG_ERROR("Validation failed.");
        }
    }

    void validate_with_our_cpu_baseline(int ans) {
        LOG_INFO("Running our baseline implementation.");
        load_graph_data_from_file();
        run_our_baseline_test();
        
        if (total_count == ans) {
            LOG_INFO("Validation passed.");
        } else {
            LOG_ERROR("Validation failed.");
        }
    }

private:
    int use_graphpi_sched = 1;
    int run_graphpi = 0;
    int run_our_baseline = 1;
    
    std::string data_name = "Wiki-Vote";
    std::string data_path = "";
    char* adj_mat;
    int pattern_size = 3;
    int patternID = 1;
    const int* p_adj_mat;
    int local_patternId = 1;
    std::string output_path = "/home/wuzhenlin/workspace/2-graphmining/X-GMiner/scripts/";
    int vertices, edges;
    std::vector<std::set<int>> edgeLists;
    long long total_count = 0;
    std::vector<int> embedding;
    Input_FileFormat file_format = Input_FileFormat::SNAP_TXT;

    bool do_validation = false;
    int vert_induced = 0;

    CPUTimer timer;
    Kernel kernel;
};
