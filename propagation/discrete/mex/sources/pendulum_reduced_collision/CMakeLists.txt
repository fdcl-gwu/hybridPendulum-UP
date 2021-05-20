cmake_minimum_required(VERSION 3.20)
project(pendulum_plot_getc LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 14)

include_directories(${PROJECT_SOURCE_DIR}/include)
set(src_dir ${PROJECT_SOURCE_DIR}/src)

find_package(Matlab REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)

include_directories("${CUDA_INCLUDE_DIRS}")

if(OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=${OpenMP_CXX_FLAGS}")
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# compile lib
add_library(getLambda STATIC ${src_dir}/setup.cpp ${src_dir}/getLambda.cu)
set_property(TARGET getLambda PROPERTY CUDA_ARCHITECTURES 80)

add_library(getOmega STATIC ${src_dir}/setup.cpp ${src_dir}/getOmega.cu)
set_property(TARGET getOmega PROPERTY CUDA_ARCHITECTURES 80)

add_library(getFcL STATIC ${src_dir}/setup.cpp ${src_dir}/getFcL.cu)
set_property(TARGET getFcL PROPERTY CUDA_ARCHITECTURES 80)

add_library(propagate STATIC ${src_dir}/setup.cpp ${src_dir}/propagate.cu)
set_property(TARGET propagate PROPERTY CUDA_ARCHITECTURES 80)
target_link_libraries(propagate cublas)

# compile mex
matlab_add_mex(
    NAME getLambda_mex
    SRC ${src_dir}/mexLambda.cpp
    LINK_TO getLambda
    R2018a
)

matlab_add_mex(
    NAME getOmega_mex
    SRC ${src_dir}/mexOmega.cpp
    LINK_TO getOmega
    R2018a
)

matlab_add_mex(
    NAME getFcL_mex
    SRC ${src_dir}/mexFcL.cpp
    LINK_TO getFcL
    R2018a
)

matlab_add_mex(
    NAME pendulum_reduced_discrete_propagate
    SRC ${src_dir}/mexPropagate.cpp
    LINK_TO propagate
    R2018a
)

