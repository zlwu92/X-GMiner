#include <../include/graph.h>
#include <../include/dataloader.h>
#include "../include/pattern.h"
#include "../include/schedule.h"
#include "../include/common.h"
#include "../include/motif_generator.h"

#include <assert.h>
#include <iostream>
#include <string>
#include <algorithm>

int main(int argc,char *argv[]) {
   
    // int size = atoi(argv[1]);
    // char* adj_mat = argv[2];

    // Pattern pattern(size, adj_mat);
    Pattern pattern(PatternType::House);
    // const int* adj_mat = pattern.get_adj_mat_ptr();
    bool is_valid;
    std::cout << __LINE__ << std::endl;
    Schedule schedule(pattern, is_valid, 0, 0, 0, 0, 0, 0);
    std::cout << __LINE__ << std::endl;
    std::vector< std::vector< std::pair<int,int> > > restricts;
    std::cout << __LINE__ << std::endl;
    schedule.restricts_generate(schedule.get_adj_mat_ptr(), restricts);
    std::cout << __LINE__ << std::endl;
    schedule.print_schedule();
    std::cout << "restricts size: " << restricts.size() << std::endl;
    for(auto &restrict : restricts) {
        for(auto &p : restrict)
            printf("(%d,%d)",p.first,p.second);
        puts("");
    }
    return 0;
}
