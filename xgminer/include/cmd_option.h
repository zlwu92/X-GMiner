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
            ("pattern-name", "Pattern name", cxxopts::value<std::string>()->default_value(""));
            ("use-graphpi-sched", "Use GraphPi scheduler", cxxopts::value<int>()->default_value("0"))
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

    std::string algo = "donothing";

    std::string pattern_name = "";
    int use_graphpi_sched = 0;
    
private:
    int argc;
    char** argv;
    struct option* long_opt;

};
