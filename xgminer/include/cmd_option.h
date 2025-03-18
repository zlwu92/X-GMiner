#pragma once
#include <string>
#include <iostream>
#include <memory>
#include <filesystem>
#include <vector>
#include "cxxopts.hpp"
#include "argparse.hpp"
#include <getopt.h>

#define PRINT_GREEN(x) std::cout << "\033[1;32m" << x << "\033[0m" << std::endl

class Command_Option {
public:

    Command_Option(int argc, char* argv[]) : argc(argc), argv(argv) {
        std::cout << "Command Line Arguments: ";
        auto app_name = std::filesystem::path(argv[0]).filename().string();
        std::cout << "\033[1;32m" << app_name << " [" << "\033[0m";
        for (int i = 1; i < argc; i++) {
            std::cout << "\033[1;32m" << argv[i] << "\033[0m";
            if (i < argc - 1) std::cout << " ";
        }
        std::cout << "\033[1;32m" << "]" << "\033[0m" << "\n";

        // show_header();
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
            ("do-validation", "Do validation", cxxopts::value<int>()->default_value("0"));
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

        } catch (const std::exception& err) {
            std::cerr << err.what() << std::endl;
            std::cerr << program;
            std::exit(1);
        }
    }

    void show_header() {
        std::cout << "(+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++)\n";
        std::cout << "(+    ____    ____     _    _    _   _    ____      _          ____     _           _       ____   +)\n";
        std::cout << "(+   / ___|  |  _ \\   | |  | |  | \\ | |  |  __|    / \\        | ___)   | |         / \\     / ___|  +)\n";
        std::cout << "(+  | |  _   | |_) |  | |  | |  |  \\| |  | |__    / _ \\   --- | |_ 、  | |        / _ \\    \\`___   +)\n";
        std::cout << "(+  | |_| |  |  __/   | |__| |  | . `.|  |  __|  / ___ \\      | |_) |  | |___、  / ___ \\    ___)|  +)\n";
        std::cout << "(+   \\____/  |_|       \\____/   |_|\\__|  |_|    /_/   \\_\\     |____/   |_____|  /_/   \\_\\  |____/  +)\n";
        std::cout << "(+                                                                                                 +)\n";
        std::cout << "(+                      GPU-based NFA Processing Engine Using BLAS Operations                      +)\n";
        std::cout << "(+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++)\n\n";
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
    
private:
    int argc;
    char** argv;
    struct option* long_opt;

};
