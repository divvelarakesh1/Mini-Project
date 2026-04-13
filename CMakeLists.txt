cmake_minimum_required(VERSION 3.16)
project(mini_project CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ──────────────────────────────────────────────
# Third-party: Eigen
# ──────────────────────────────────────────────
add_library(Eigen INTERFACE)
target_include_directories(Eigen INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/third-party/eigen
)

# ──────────────────────────────────────────────
# Third-party: MiniDNN
# ──────────────────────────────────────────────
add_library(MiniDNN INTERFACE)
target_include_directories(MiniDNN INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/third-party/MiniDNN/include
)
target_link_libraries(MiniDNN INTERFACE Eigen)

# ──────────────────────────────────────────────
# Third-party: nlohmann/json
# ──────────────────────────────────────────────
add_library(nlohmann_json INTERFACE)
target_include_directories(nlohmann_json INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/third-party
)
# ──────────────────────────────────────────────
# Project sources
# ──────────────────────────────────────────────
set(PROJECT_SOURCES
    src/main.cpp
    src/data.cpp
    src/worker.cpp
    src/monitor.cpp
    src/engine.cpp
    src/config.cpp
    src/optimizer.cpp
)

add_executable(mini_project ${PROJECT_SOURCES})

# Include Directories
target_include_directories(mini_project PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src 
)

# Link basic libraries
target_link_libraries(mini_project PRIVATE Eigen MiniDNN nlohmann_json)

# ──────────────────────────────────────────────
# The Apple Silicon Bulletproof OpenMP Fix
# ──────────────────────────────────────────────
if(APPLE)
    # 1. Ask Mac exactly where Homebrew lives
    execute_process(COMMAND brew --prefix OUTPUT_VARIABLE HOMEBREW_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
    
    # 2. Force AppleClang to use OpenMP directly (Bypass find_package completely)
    target_compile_options(mini_project PRIVATE -Xclang -fopenmp)
    target_include_directories(mini_project PRIVATE ${HOMEBREW_PREFIX}/opt/libomp/include)
    target_link_options(mini_project PRIVATE -lomp -L${HOMEBREW_PREFIX}/opt/libomp/lib)
else()
    # Standard Linux/Windows behavior
    find_package(OpenMP REQUIRED)
    target_link_libraries(mini_project PRIVATE OpenMP::OpenMP_CXX)
endif()

# ──────────────────────────────────────────────
# Compiler Optimizations for ML Training
# ──────────────────────────────────────────────
target_compile_options(mini_project PRIVATE
    -Wall -Wextra 
    -O3           
)