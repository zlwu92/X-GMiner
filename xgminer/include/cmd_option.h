#pragma once
#include <string>
#include <iostream>
#include <memory>
#include <filesystem>
#include <vector>
#include "cxxopts.hpp"
#include "argparse.hpp"
#include <getopt.h>
#include "utils.h"

#define PRINT_GREEN(x) std::cout << "\033[1;36m" << x << "\033[0m" << std::endl

class Command_Option {
public:

    Command_Option(int argc, char* argv[]) : argc(argc), argv(argv) {
        std::cout << "Command Line Arguments: ";
        auto app_name = std::filesystem::path(argv[0]).filename().string();
        std::cout << "\033[1;34m" << app_name << " [" << "\033[0m";
        for (int i = 1; i < argc; i++) {
            std::cout << "\033[1;34m" << argv[i] << "\033[0m";
            if (i < argc - 1) std::cout << " ";
        }
        std::cout << "\033[1;34m" << "]" << "\n";
        std::cout << "-------------------------------------------------------------------------------------\033[0m";
        show_header();
    }

    void parseByCxxOpts() {
        auto app_name = std::filesystem::path(argv[0]).filename().string();
        cxxopts::Options options(app_name, "Command Line Options");
        options.positional_help("[optional args]").show_positional_help();
        options.add_options()
            ("h,help", "Print usage")
            ("g,graph", "Graph file path", cxxopts::value<std::string>()->default_value(""))
            ("algorithm", "Algorithm", cxxopts::value<std::string>()->default_value("donothing"))
            ("pattern-name", "Pattern name", cxxopts::value<std::string>()->default_value(""))
            ("use-graphpi-sched", "Use GraphPi scheduler", cxxopts::value<int>()->default_value("0"))
            ("d,dataname", "Data name", cxxopts::value<std::string>()->default_value("Wiki-Vote"))
            ("run-graphpi", "Run GraphPi algorithm", cxxopts::value<int>()->default_value("0"))
            ("run-our-baseline", "Run our baseline implementation", cxxopts::value<int>()->default_value("0"))
            ("pattern-adj-mat", "Pattern adjacency matrix", cxxopts::value<std::string>()->default_value("011101110"))
            ("pattern-size", "Pattern size", cxxopts::value<int>()->default_value("3"))
            ("patternID", "Pattern ID", cxxopts::value<int>()->default_value("1"))
            ("do-validation", "Do validation", cxxopts::value<int>()->default_value("0"))
            ("vert-induced", "Vertex induced", cxxopts::value<int>()->default_value("0"))
            ("tunek", "Tune k", cxxopts::value<int>()->default_value("0"))
            ("setk", "Set k", cxxopts::value<int>()->default_value("0"))
            ("run-xgminer", "Run X-GMiner", cxxopts::value<int>()->default_value("0"))
            ("prof-workload", "Profile workload", cxxopts::value<int>()->default_value("0"))
            ("use-vert-para", "Use vertex parallelism", cxxopts::value<int>()->default_value("0"))
            ("prof-edgecheck", "Profile edge check", cxxopts::value<int>()->default_value("0"));
        ;
        
        try {
            auto result = options.parse(argc, argv);
            
            if (result.count("help")) {
                std::cout << options.help() << std::endl;
                exit(0);
            }

            if (result.count("graph")) {
                datagraph_file = result["graph"].as<std::string>();
                PRINT_GREEN("Data graph: " << datagraph_file);
            }

            if (result.count("algorithm")) {
                algo = result["algorithm"].as<std::string>();
                PRINT_GREEN("Algo: " << algo);
            }

            if (result.count("pattern-name")) {
                pattern_name = result["pattern-name"].as<std::string>();
                PRINT_GREEN("Pattern name: " << pattern_name);
            }

            if (result.count("use-graphpi-sched")) {
                use_graphpi_sched = result["use-graphpi-sched"].as<int>();
                PRINT_GREEN("Use GraphPi scheduler: " << use_graphpi_sched);
            }

            if (result.count("dataname")) {
                data_name = result["dataname"].as<std::string>();
                PRINT_GREEN("Data name: " << data_name);
            }

            if (result.count("run-graphpi")) {
                run_graphpi = result["run-graphpi"].as<int>();
                PRINT_GREEN("Run GraphPi algorithm: " << run_graphpi);
            }

            if (result.count("run-our-baseline")) {
                run_our_baseline = result["run-our-baseline"].as<int>();
                PRINT_GREEN("Run our baseline implementation: " << run_our_baseline);
            }

            if (result.count("pattern-adj-mat")) {
                pattern_adj_mat = strdup(result["pattern-adj-mat"].as<std::string>().c_str());
                // pattern_adj_mat = result["pattern-adj-mat"].as<std::string>().c_str();
                PRINT_GREEN("Pattern adjacency matrix: " << pattern_adj_mat);
            }

            if (result.count("pattern-size")) {
                pattern_size = result["pattern-size"].as<int>();
                PRINT_GREEN("Pattern size: " << pattern_size);
            }

            if (result.count("patternID")) {
                patternID = result["patternID"].as<int>();
                PRINT_GREEN("Pattern ID: " << patternID);
            }

            if (result.count("do-validation")) {
                do_validation = result["do-validation"].as<int>();
                PRINT_GREEN("Do validation: " << do_validation);
            }

            if (result.count("vert-induced")) {
                vert_induced = result["vert-induced"].as<int>();
                PRINT_GREEN("Vertex induced: " << vert_induced);
            }

            if (result.count("tunek")) {
                tune_k = result["tunek"].as<int>();
                PRINT_GREEN("Tune k: " << tune_k);
            }

            if (result.count("setk")) {
                set_k = result["setk"].as<int>();
                PRINT_GREEN("Set k: " << set_k);
            }

            if (result.count("run-xgminer")) {
                run_xgminer = result["run-xgminer"].as<int>();
                PRINT_GREEN("Run X-GMiner: " << run_xgminer);
            }

            if (result.count("prof-workload")) {
                prof_workload = result["prof-workload"].as<int>();
                PRINT_GREEN("Profile workload: " << prof_workload);
            }

            if (result.count("use-vert-para")) {
                use_vert_para = result["use-vert-para"].as<int>();
                PRINT_GREEN("Use vertex parallelism: " << use_vert_para);
            }

            if (result.count("prof-edgecheck")) {
                prof_edgecheck = result["prof-edgecheck"].as<int>();
                PRINT_GREEN("Profile edge check: " << prof_edgecheck);
            }

        } catch (const cxxopts::exceptions::exception& e) {
            std::cerr << "Error parsing options: " << e.what() << std::endl;
            exit(1);
        }
    }

    void parseByArgParse() {
        auto app_name = std::filesystem::path(argv[0]).filename().string();
        argparse::ArgumentParser program(app_name, "Command Line Options");
        
        program.add_argument("-g", "--graph").help("Input graph file path").default_value(std::string{""});
        program.add_argument("--algorithm").help("Algorithm").default_value(std::string{"donothing"});
        program.add_argument("--pattern-name").help("Pattern name").default_value(std::string{""});
        program.add_argument("--use-graphpi-sched").help("Use GraphPi scheduler").default_value(0).action(
                                                        [](const std::string& value) { return std::stoi(value); });
        program.add_argument("-d", "--dataname").help("Data name").default_value(std::string{"Wiki-Vote"});
        program.add_argument("--run-graphpi").help("Run GraphPi algorithm").default_value(0).action(
                                                        [](const std::string& value) { return std::stoi(value); });
        program.add_argument("--run-our-baseline").help("Run our baseline implementation").default_value(0).action(
                                                        [](const std::string& value) { return std::stoi(value); });
        program.add_argument("--pattern-adj-mat").help("Pattern adjacency matrix").default_value(std::string{"011101110"});
        program.add_argument("--pattern-size").help("Pattern size").default_value(3).action(
                                                        [](const std::string& value) { return std::stoi(value); });
        program.add_argument("--patternID").help("Pattern ID").default_value(1).action(
                                                        [](const std::string& value) { return std::stoi(value); });
        program.add_argument("--do-validation").help("Do validation").default_value(0).action(
                                                        [](const std::string& value) { return std::stoi(value); });
        program.add_argument("--vert-induced").help("Vertex induced").default_value(0).action(
                                                        [](const std::string& value) { return std::stoi(value); });
        program.add_argument("--tunek").help("Tune k").default_value(0).action(
                                                        [](const std::string& value) { return std::stoi(value); });
        program.add_argument("--setk").help("Set k").default_value(0).action(
                                                        [](const std::string& value) { return std::stoi(value); });
        program.add_argument("--run-xgminer").help("Run X-GMiner").default_value(0).action(
                                                        [](const std::string& value) { return std::stoi(value); });
        program.add_argument("--prof-workload").help("Profile workload").default_value(0).action(
                                                        [](const std::string& value) { return std::stoi(value); });
        program.add_argument("--use-vert-para").help("Use vertex parallelism").default_value(0).action(
                                                        [](const std::string& value) { return std::stoi(value); });
        program.add_argument("--prof-edgecheck").help("Profile edge check").default_value(0).action(
                                                        [](const std::string& value) { return std::stoi(value); });

        try {
            program.parse_args(argc, argv);

            if (program.get<std::string>("--graph") != "") {
                datagraph_file = program.get<std::string>("--graph");
                PRINT_GREEN("Data graph: " << datagraph_file);
            }

            if (program.get<std::string>("--algorithm") != "") {
                algo = program.get<std::string>("--algorithm");
                PRINT_GREEN("Algo: " << algo);
            }

            if (program.get<std::string>("--pattern-name") != "") {
                pattern_name = program.get<std::string>("--pattern-name");
                PRINT_GREEN("Pattern name: " << pattern_name);
            }

            if (program.get<int>("--use-graphpi-sched") != 0) {
                use_graphpi_sched = program.get<int>("--use-graphpi-sched");
                PRINT_GREEN("Use GraphPi scheduler: " << use_graphpi_sched);
            }

            if (program.get<std::string>("--dataname") != "") {
                data_name = program.get<std::string>("--dataname");
                PRINT_GREEN("Data name: " << data_name);
            }

            if (program.get<int>("--run-graphpi") != 0) {
                run_graphpi = program.get<int>("--run-graphpi");
                PRINT_GREEN("Run GraphPi algorithm: " << run_graphpi);
            }

            if (program.get<int>("--run-our-baseline") != 0) {
                run_our_baseline = program.get<int>("--run-our-baseline");
                PRINT_GREEN("Run our baseline implementation: " << run_our_baseline);
            }

            if (program.get<std::string>("--pattern-adj-mat") != "") {
                printf("heree@!!!!\n");
                std::cout << program.get<std::string>("--pattern-adj-matrix") << std::endl;
                pattern_adj_mat = strdup(program.get<std::string>("--pattern-adj-matrix").c_str());
                PRINT_GREEN("Pattern adjacency matrix: " << pattern_adj_mat);
            }

            if (program.get<int>("--pattern-size") != 0) {
                pattern_size = program.get<int>("--pattern-size");
                PRINT_GREEN("Pattern size: " << pattern_size);
            }

            if (program.get<int>("--patternID") != 0) {
                patternID = program.get<int>("--patternID");
                PRINT_GREEN("Pattern ID: " << patternID);
            }

            if (program.get<int>("--do-validation") != 0) {
                do_validation = program.get<int>("--do-validation");
                PRINT_GREEN("Do validation: " << do_validation);
            }

            if (program.get<int>("--vert-induced") != 0) {
                vert_induced = program.get<int>("--vert-induced");
                PRINT_GREEN("Vertex induced: " << vert_induced);
            }

            if (program.get<int>("--tunek") != 0) {
                tune_k = program.get<int>("--tunek");
                PRINT_GREEN("Tune k: " << tune_k);
            }

            if (program.get<int>("--setk") != 0) {
                set_k = program.get<int>("--setk");
                PRINT_GREEN("Set k: " << set_k);
            }

            if (program.get<int>("--run-xgminer") != 0) {
                run_xgminer = program.get<int>("--run-xgminer");
                PRINT_GREEN("Run X-GMiner: " << run_xgminer);
            }

            if (program.get<int>("--prof-workload") != 0) {
                prof_workload = program.get<int>("--prof-workload");
                PRINT_GREEN("Profile workload: " << prof_workload);
            }

            if (program.get<int>("--use-vert-para") != 0) {
                use_vert_para = program.get<int>("--use-vert-para");
                PRINT_GREEN("Use vertex parallelism: " << use_vert_para);
            }

            if (program.get<int>("--prof-edgecheck") != 0) {
                prof_edgecheck = program.get<int>("--prof-edgecheck");
                PRINT_GREEN("Profile edge check: " << prof_edgecheck);
            }

        } catch (const std::exception& err) {
            std::cerr << err.what() << std::endl;
            std::cerr << program;
            std::exit(1);
        }
    }

    void show_header() {
        const std::string xgminer_logo = R"(
(++++++++++++++++++++++++++++++++++++++++++++++++++)
(+   __  __      ____ __  __ _                    +)
(+   \ \/ /     / ___|  \/  (_)_ __   ___ _ __    +)
(+    \  /_____| |  _| |\/| | | '_ \ / _ \ '__|   +)
(+    /  \_____| |_| | |  | | | | | |  __/ |      +)
(+   /_/\_\     \____|_|  |_|_|_| |_|\___|_|      +)
(+                                                +)
(+      GPU-accelerated Graph Mining Engine       +)
)";
        std::cout << "\033[1;32m" << xgminer_logo;
        // std::cout << "Author: Zhenlin Wu" << std::endl;
        std::time_t now = std::time(nullptr);
        std::tm* local_time = std::localtime(&now);
        char buffer[80];
        std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", local_time);
        std::cout << "(+  X-GMiner version: v0.1 @ " << buffer << "  +)" << std::endl;
        std::cout << "(++++++++++++++++++++++++++++++++++++++++++++++++++)\033[0m\n";
        // std::cout << "Description: A GPU-accelerated graph mining engine for pattern matching." << std::endl;
    }

    std::string datagraph_file = "";
    std::string data_name = "Wiki-Vote";
    std::string algo = "donothing";

    std::string pattern_name = "";
    int use_graphpi_sched = 0;
    int run_graphpi = 0;
    int run_our_baseline = 0;
    char* pattern_adj_mat = nullptr;
    int pattern_size = 3;
    int patternID = 1;
    int do_validation = 0;
    int vert_induced = 0;
    int tune_k = 0; // control bitmap bucket number
    int set_k = 0; // set a specific bitmap bucket number mannually
    int run_xgminer = 0;
    int prof_workload = 0;
    int use_vert_para = 0;
    int prof_edgecheck = 0;
    
private:
    int argc;
    char** argv;
    struct option* long_opt;

};
