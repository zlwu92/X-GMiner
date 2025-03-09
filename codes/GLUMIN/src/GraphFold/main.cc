#include "graph.h"
#include "pattern.hh"

void PatternSolver(Graph &g, int k, std::vector<uint64_t> &accum, int, int);
void CliqueSolver(Graph &g, int k, uint64_t &total, int, int);

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cout << "Usage: " << argv[0] << "<graph> <k> [ngpu(0)] [chunk_size(1024)]\n";
    std::cout << "Example: " << argv[0] << " /inputs/mico/graph P4 lut\n";
    exit(1);
  }
  std::string pattern_name = argv[2];
  std::regex pattern("^P(\\d+)$");
  std::smatch match;
  int k;
  if (std::regex_match(pattern_name, match, pattern)) {
    k = std::stoi(match[1].str());
    std::cout << "Pattern P" << k << std::endl;
  } else {
    std::cerr << "Invalid input format. Expected format: Px, where x is an integer." << std::endl;
    return 1;
  }
  std::cout << "P" << k << "(only for undirected graphs)\n";
  std::string use_lut;
  if (argc > 3) use_lut = argv[3];
  int k_num = k;
  if (use_lut != "lut") k_num = k + 1;
  
  if (k_num == 5 || k_num == 6) {
    Graph g(argv[1], USE_DAG); // use DAG
    int n_devices = 8;
    int chunk_size = 1024;
    int select_device = 3;
    int choose = 0;
    if (argc > 4) select_device = atoi(argv[4]);
    if (argc > 5) n_devices = atoi(argv[5]);
    if (argc > 6) chunk_size = atoi(argv[6]);
    g.print_meta_data();
  
    uint64_t total = 0;
    CliqueSolver(g, k_num, total, select_device, n_devices);
    std::cout << "Pattern P" << k << " count: " << total << "\n";
  }
  else {
    Graph g(argv[1]);
    int n_devices = 1;
    int chunk_size = 1024;
    if (argc > 4) n_devices = atoi(argv[4]);
    if (argc > 5) chunk_size = atoi(argv[5]);
    g.print_meta_data();

    int num_patterns = 1;
    std::cout << "num_patterns: " << num_patterns << "\n";
    std::vector<uint64_t> total(num_patterns, 0);
    PatternSolver(g, k_num, total, n_devices, chunk_size);
    for (int i = 0; i < num_patterns; i++)
      std::cout << "Pattern P" << k << " count: " << total[i] << "\n";
    return 0;
  }
}

