# 使用 ExternalProject 管理外部 Makefile 项目
# include(ExternalProject)
# ExternalProject_Add(
#     GLUMIN
#     SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/GLUMIN
#     CONFIGURE_COMMAND ""  # 跳过配置步骤
#     BUILD_COMMAND make -C ${CMAKE_CURRENT_SOURCE_DIR}/GLUMIN all  # 直接调用 make
#     INSTALL_COMMAND ""   # 跳过安装步骤
#     BUILD_ALWAYS ON      # 每次构建时都重新编译
#     COMMENT "Running Makefile in GLUMIN"
# )

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/GLUMIN)
add_custom_target(glumin ALL
    COMMAND make -C ${CMAKE_CURRENT_SOURCE_DIR}/GLUMIN all
    COMMENT "Running Makefile in GLUMIN"
)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/GraphMiner)
add_custom_target(g2miner ALL
    COMMAND make -C ${CMAKE_CURRENT_SOURCE_DIR}/GraphMiner all
    COMMENT "Running Makefile in GraphMiner"
)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/GraphFold GraphFold)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/GraphPi GraphPi)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/GraphSet GraphSet)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/HEROFramework HERO)
