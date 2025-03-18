/*
    * common.cpp
    *
    * borrowed from GraphPi and GLUMIN
*/

#include "common.h"
#include <sys/time.h>
#include <cstdlib>
#include <filesystem>
#include <string>

/*********************************** GraphPi *******************************/
double get_wall_time() {
    struct timeval time;
    if(gettimeofday(&time,NULL)) {
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * 0.000001;
}

int read_int() {
    char ch = getchar();
    while((ch < '0' || ch > '9') && ch !='-') ch = getchar();
    int tag = 1;
    if(ch == '-') tag = -1, ch = getchar();
    int x = 0;
    while( ch >= '0' && ch <= '9') x = x* 10 + ch -'0', ch = getchar();
    return x * tag;
}

unsigned int read_unsigned_int() {
    char ch = getchar();
    while((ch < '0' || ch > '9') ) ch = getchar();
    unsigned int x = 0;
    while( ch >= '0' && ch <= '9') x = x* 10 + ch -'0', ch = getchar();
    return x;
}


/*********************************** X-GMiner *******************************/
namespace fs = std::filesystem;


bool checkFileExtension(std::string filename, const std::string& extension) {
    if (filename.length() < extension.length() + 1) {
        return false;
    }

    std::string fileExtension = filename.substr(filename.length() - extension.length());
    std::transform(fileExtension.begin(), fileExtension.end(), fileExtension.begin(), ::tolower);

    return fileExtension == extension;
}

bool checkDirectoryForFilesWithExtension(std::string directoryPath, const std::string& extension) {
    for (auto& entry : fs::directory_iterator(directoryPath)) {
        if (fs::is_regular_file(entry) && checkFileExtension(entry.path().filename(), extension)) {
            std::cout << "Found file with extension '" << extension << "': " << entry.path() << std::endl;
            return true;
        }
    }
    return false;
}

Input_FileFormat getFileFormat(std::string filename) {
    printf("Checking file format of %s\n", filename.c_str());
    if (fs::is_directory(filename)) {
        printf("Directory\n");
        if (checkDirectoryForFilesWithExtension(filename, "bin")) {
            return BINARY; // for GLUMIN and G2Miner
        }
    } else if (fs::is_regular_file(filename)) {
        if (checkFileExtension(filename, "txt")) {
            return SNAP_TXT; // for GraphPi
        } else if (checkFileExtension(filename, "mtx")) {
            return MatrixMarket; // for GraphFold
        } 
        else if (!fs::path(filename).has_extension()) {
            std::ifstream file(filename);
            std::string line;
            bool firstLineContainsTwoNumbers = false;
            if (std::getline(file, line)) {
                std::istringstream iss(line);
                int num1, num2;
                if (iss >> num1 >> num2 && iss.eof()) {
                    char extraCheck;
                    if (!(iss >> extraCheck)) {
                        firstLineContainsTwoNumbers = true;
                    }
                }
            }
            file.close();
            if (firstLineContainsTwoNumbers)    return SNAP_TXT;
            else   return INVALID;
        }
        else {
            return INVALID;
        }
    }

    return INVALID;
}
