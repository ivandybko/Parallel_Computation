cmake_minimum_required(VERSION 3.30)
project(Lab_3)

set(CMAKE_CXX_STANDARD 20)
set(MPI_HOME "/usr/local/openmpi")
set(ENV{MPI_HOME} ${MPI_HOME})

find_package(MPI REQUIRED)

set(CMAKE_C_COMPILER "/opt/homebrew/opt/llvm/bin/clang")
set(CMAKE_CXX_COMPILER "/opt/homebrew/opt/llvm/bin/clang++")

include_directories("/opt/homebrew/opt/libomp/include")
link_directories("/opt/homebrew/opt/libomp/lib")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lomp -L/opt/homebrew/opt/libomp/lib")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fopenmp -march=native -funroll-loops -fomit-frame-pointer -DNDEBUG")

add_executable(Lab_3 Helmholtz_Dybko.cpp)
