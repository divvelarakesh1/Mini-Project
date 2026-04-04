cmake_minimum_required(VERSION 3.16)
project(MiniProject CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ──────────────────────────────────────────────
# Third-party: Eigen (header-only)
# Usage:  #include <Eigen/Dense>  etc.
# ──────────────────────────────────────────────
add_library(Eigen INTERFACE)
target_include_directories(Eigen INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/third-party/eigen
)

# ──────────────────────────────────────────────
# Third-party: MiniDNN (header-only)
# Usage:  #include <MiniDNN.h>  etc.
# MiniDNN itself needs Eigen, so we link them.
# ──────────────────────────────────────────────
add_library(MiniDNN INTERFACE)
target_include_directories(MiniDNN INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/third-party/MiniDNN/include
)
target_link_libraries(MiniDNN INTERFACE Eigen)

# ──────────────────────────────────────────────
# OpenMP (optional – install with: brew install libomp)
# ──────────────────────────────────────────────
find_package(OpenMP)
if(NOT OpenMP_CXX_FOUND)
    message(WARNING "OpenMP not found – parallel pragmas will be disabled. Run: brew install libomp")
endif()

# ──────────────────────────────────────────────
# Project sources
# ──────────────────────────────────────────────
set(PROJECT_SOURCES
    src/main.cpp
    src/data.cpp
    src/worker.cpp
    src/monitor.cpp
)

add_executable(mini_project ${PROJECT_SOURCES})

# Project's own headers (include/)
target_include_directories(mini_project PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Link third-party INTERFACE libraries + OpenMP if available
target_link_libraries(mini_project PRIVATE
    Eigen
    MiniDNN
    $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>
)

# Optional: enable warnings
target_compile_options(mini_project PRIVATE
    -Wall -Wextra -O2
)
