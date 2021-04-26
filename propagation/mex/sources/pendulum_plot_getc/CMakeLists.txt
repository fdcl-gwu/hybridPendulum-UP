cmake_minimum_required(VERSION 3.20)
project(pendulum_plot_getc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(/home/wwang442/fftw-3.3.9/api)
set(src_dir ${PROJECT_SOURCE_DIR}/src)

find_package(Matlab REQUIRED)
find_package(OpenMP REQUIRED)

if(OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# compile lib
add_library(getc STATIC ${src_dir}/getc.cpp ${src_dir}/setup.cpp)
target_link_libraries(getc fftw3)

# compile mex
matlab_add_mex(
    NAME pendulum_plot_getc
    SRC ${src_dir}/mexFunction.cpp
    LINK_TO getc
    R2018a
)

