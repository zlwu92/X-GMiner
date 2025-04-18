cmake_minimum_required(VERSION 3.12)

# project(xgminer LANGUAGES CXX CUDA)
project(xgminer)
enable_language(CXX CUDA)

find_package(OpenMP REQUIRED)
#find_package(nlohmann_json REQUIRED)

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

set(DEFAULT_TEMPLATE_PARAM_BMAP_BUCKET_NUM 4) # k

if(NOT DEFINED BMAP_BUCKET_NUM)
    set(BMAP_BUCKET_NUM ${DEFAULT_TEMPLATE_PARAM_BMAP_BUCKET_NUM} CACHE STRING "Template parameter for CUDA kernel")
endif()

# 将模板参数值传递给编译器
add_definitions(-DBMAP_BUCKET_NUM=${BMAP_BUCKET_NUM})

# --ptxas-options=-v
set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -O3 -lineinfo --expt-extended-lambda -Wno-deprecated-gpu-targets \
                    -Wall -Wno-deprecated -arch=compute_86 --disable-warnings")
# -Wno-unused-variable

include_directories(
    "${PROJECT_SOURCE_DIR}/include/" 
    "${PROJECT_SOURCE_DIR}/include/graphpi/" 
    "${PROJECT_SOURCE_DIR}/include/glumin/"
    "${PROJECT_SOURCE_DIR}/src/gpu_kernels/"
    "${PROJECT_SOURCE_DIR}/src/gpu_kernels/glumin/"
    "${PROJECT_SOURCE_DIR}/src/gpu_kernels/glumin/gpu_GM_LUT_kernels/"
    "${PROJECT_SOURCE_DIR}/src/gpu_kernels/xgminer/"
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
    "${PROJECT_SOURCE_DIR}/src/glumin/*.cu"
    "${PROJECT_SOURCE_DIR}/src/glumin/*.cpp"
    # "${PROJECT_SOURCE_DIR}/src/xgminer/*.cu"
    # "${PROJECT_SOURCE_DIR}/src/xgminer/*.cpp"
    "${PROJECT_SOURCE_DIR}/src/gpu_kernels/*.cpp"
    "${PROJECT_SOURCE_DIR}/src/gpu_kernels/*.cu"
    "${PROJECT_SOURCE_DIR}/src/gpu_kernels/glumin/gpu_GM_LUT_kernels/*.cu"
)

# 合并源文件列表
list(APPEND SOURCES ${SRC_FILES})

# file(GLOB_RECURSE SOURCES RELATIVE ${PROJECT_SOURCE_DIR} main.cpp "src/*" "src/graphpi/*")
# cuda_add_executable(xgminer ${SOURCES})
add_executable(xgminer ${SOURCES})

# 使用 target_include_directories 替代 include_directories（现代 CMake 推荐）
# 添加目标并递归包含 src/gpu_kernels/ 及其子目录
# target_include_directories(xgminer PRIVATE
#     "${PROJECT_SOURCE_DIR}/include/"
#     "${PROJECT_SOURCE_DIR}/include/graphpi/"
#     "${PROJECT_SOURCE_DIR}/include/glumin/"
#     "${PROJECT_SOURCE_DIR}/src/gpu_kernels/"
#     ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
# )

set(ONEAPI_SDK_HOME "/opt/intel/oneapi/vtune/2025.0/sdk/")
#target_include_directories(xgminer PUBLIC "${ONEAPI_SDK_HOME}/include")
#target_link_directories(xgminer PUBLIC "${ONEAPI_SDK_HOME}/lib64")
#target_link_libraries(xgminer PUBLIC ittnotify ${CMAKE_DL_LIBS})

set_target_properties(
    xgminer PROPERTIES CUDA_ARCHITECTURES "86;86"
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED OFF
    CXX_EXTENSIONS OFF
    CUDA_STANDARD 17
)

target_link_libraries(xgminer PRIVATE 
    OpenMP::OpenMP_CXX 
    #    nlohmann_json::nlohmann_json
    ${MPI_CXX_LIBRARIES}
)


# 指定特定版本的 CUDA 工具链
# set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-11.7")
# set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc")

# # 添加 CUDA 库路径和头文件路径
# list(APPEND CMAKE_PREFIX_PATH "${CUDA_TOOLKIT_ROOT_DIR}")
# include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)
# link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)

# # 检查 CUDA 是否正确找到
# find_package(CUDA REQUIRED)

# # 设置目标 GPU 架构
# set(CMAKE_CUDA_ARCHITECTURES 75)  # 示例：支持 Turing 架构 (sm_75)

# # 添加可执行文件
# add_executable(cuda_cxx_example main.cu helper.cpp)

# # 链接 CUDA 运行时库
# target_link_libraries(cuda_cxx_example PRIVATE cudart)