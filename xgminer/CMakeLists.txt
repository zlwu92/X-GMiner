cmake_minimum_required(VERSION 3.12)

# project(xgminer LANGUAGES CXX CUDA)
project(xgminer)
enable_language(CXX CUDA)

find_package(OpenMP REQUIRED)
find_package(nlohmann_json REQUIRED)

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -Wno-unused-result") # -fopenmp
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package(MPI REQUIRED)
if(MPI_FOUND)
    SET(CMAKE_CXX_COMPILER mpicxx)   
endif()

set(CMAKE_CUDA_ARCHITECTURES 86)

# Set CUDA architectures
set(CUDA_ARCHITECTURES "86")

option(DEBUG_ENABLED "Enable debug mode" OFF)
if (DEBUG_ENABLED)
    add_compile_definitions(DEBUG)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")  # 调试模式添加 CUDA 调试符号
endif()

add_definitions(-DTHRUST_IGNORE_DEPRECATED_CPP_DIALECT)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

# --ptxas-options=-v
set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -O3 -lineinfo --expt-extended-lambda -Wno-deprecated-gpu-targets \
                    -Wall -Wno-deprecated -arch=compute_86 --disable-warnings")
# -Wno-unused-variable

include_directories(
    "${PROJECT_SOURCE_DIR}/include/" 
    "${PROJECT_SOURCE_DIR}/include/graphpi/" 
    "${PROJECT_SOURCE_DIR}/include/glumin/"
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

message(STATUS "CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES: ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
message(STATUS "CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")
message(STATUS "PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")

set(SOURCES main.cpp)
# 递归收集 src/ 和 src/graphpi/ 下的所有 .cpp 文件
file(GLOB_RECURSE SRC_FILES 
    "${PROJECT_SOURCE_DIR}/src/*.cpp"
    "${PROJECT_SOURCE_DIR}/src/*.cu"
    "${PROJECT_SOURCE_DIR}/src/graphpi/*.cpp"
    "${PROJECT_SOURCE_DIR}/src/cpu_kernels/*.cpp"
    "${PROJECT_SOURCE_DIR}/src/gpu_kernels/*.cpp"
    "${PROJECT_SOURCE_DIR}/src/gpu_kernels/*.cu"
)

# 合并源文件列表
list(APPEND SOURCES ${SRC_FILES})

# file(GLOB_RECURSE SOURCES RELATIVE ${PROJECT_SOURCE_DIR} main.cpp "src/*" "src/graphpi/*")
# cuda_add_executable(xgminer ${SOURCES})
add_executable(xgminer ${SOURCES})

# 使用 target_include_directories 替代 include_directories（现代 CMake 推荐）
# target_include_directories(xgminer PRIVATE
#     ${PROJECT_SOURCE_DIR}/include       # 包含 include/ 目录
#     ${PROJECT_SOURCE_DIR}/include/graphpi  # 包含 include/graphpi/
#     ${CUDA_INCLUDE_DIRS}                # 包含 CUDA 头文件
# )

set_target_properties(
    xgminer PROPERTIES CUDA_ARCHITECTURES "86;86"
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED OFF
    CXX_EXTENSIONS OFF
    CUDA_STANDARD 17
)

target_link_libraries(xgminer PRIVATE 
    OpenMP::OpenMP_CXX 
    nlohmann_json::nlohmann_json
    ${MPI_CXX_LIBRARIES}
)
