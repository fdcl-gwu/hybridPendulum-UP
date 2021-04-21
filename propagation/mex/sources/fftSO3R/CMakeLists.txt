cmake_minimum_required(VERSION 3.15)
project(fftSO3R LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
find_package(Matlab)
find_package(CUDA REQUIRED)
set(CMAKE_CUDA_STANDARD 14)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")

set(CMAKE_CXX_STANDARD 11)

matlab_add_mex(
    NAME fftSO3R
    SRC kernel.cu setup.cu
    LINK_TO cufft
    R2018a
)

set_property(TARGET fftSO3R PROPERTY CUDA_ARCHITECTURES 80)

include_directories(~/libcutensor/include)

target_link_libraries(fftSO3R ~/libcutensor/lib/11/libcutensor.so)

