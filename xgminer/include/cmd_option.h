#ifndef CMD_OPTION_H_ 
#define CMD_OPTION_H_
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
            ("a,automata", "Automata file path", cxxopts::value<std::string>()->default_value(""))
            ("i,input", "Input file path", cxxopts::value<std::string>()->default_value(""))
            ("algorithm", "Algorithm", cxxopts::value<std::string>()->default_value("donothing"))
            ("testing-input", "Testing input file path", cxxopts::value<std::string>()->default_value(""))
            ("start-of-input", "Start of input", cxxopts::value<int>()->default_value("-1"))
            ("length-of-input", "Length of input", cxxopts::value<int>()->default_value("-1"))
        ;
        
        try {
            auto result = options.parse(argc, argv);
            
            if (result.count("help")) {
                std::cout << options.help() << std::endl;
                exit(0);
            }

            if (result.count("automata")) {
                automata_filename = result["automata"].as<std::string>();
                PRINT_GREEN("Automata: " << automata_filename);
            }

            if (result.count("input")) {
                input_filename = result["input"].as<std::string>();
                PRINT_GREEN("Input stream: " << input_filename);
            }

            if (result.count("algorithm")) {
                algo = result["algorithm"].as<std::string>();
                PRINT_GREEN("Algo: " << algo);
            }

            if (result.count("length-of-input")) {
                length_of_input = result["length-of-input"].as<int>();
                PRINT_GREEN("Length of input: " << length_of_input);
            }

            

        } catch (const cxxopts::exceptions::exception& e) {
            std::cerr << "Error parsing options: " << e.what() << std::endl;
            exit(1);
        }
    }

    void parseByArgParse() {
        auto app_name = std::filesystem::path(argv[0]).filename().string();
        argparse::ArgumentParser program(app_name, "Command Line Options");
        
        program.add_argument("-a", "--automata").help("Input file path").default_value(std::string{""});
        program.add_argument("-i", "--input").help("Automata file path").default_value(std::string{""});
        program.add_argument("--algorithm").help("Algorithm").default_value(std::string{"donothing"});
        program.add_argument("--testing-input").help("Testing input file path").default_value(std::string{""});
        program.add_argument("--start-of-input").help("Start of input").default_value(-1).action(
                                                        [](const std::string& value) { return std::stoi(value); });
        program.add_argument("--length-of-input").help("Length of input").default_value(-1).action(
                                                        [](const std::string& value) { return std::stoi(value); });
        
        try {
            program.parse_args(argc, argv);

            if (program.get<std::string>("--automata") != "") {
                automata_filename = program.get<std::string>("--automata");
                PRINT_GREEN("Automata: " << automata_filename);
            }

            if (program.get<std::string>("--input") != "") {
                input_filename = program.get<std::string>("--input");
                PRINT_GREEN("Input stream: " << input_filename);
            }

            if (program.get<std::string>("--algorithm") != "") {
                algo = program.get<std::string>("--algorithm");
                PRINT_GREEN("Algo: " << algo);
            }

            if (program.get<int>("--length-of-input") != -1) {
                length_of_input = program.get<int>("--length-of-input");
                PRINT_GREEN("Length of input: " << length_of_input);
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


    
private:
    int argc;
    char** argv;
    struct option* long_opt;

};

#endif
