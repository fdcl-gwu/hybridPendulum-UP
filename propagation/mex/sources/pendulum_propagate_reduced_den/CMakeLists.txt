cmake_minimum_required(VERSION 3.15)
project(pendulum_propagate LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
find_package(Matlab)
find_package(CUDA REQUIRED)
set(CMAKE_CUDA_STANDARD 14)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")

set(CMAKE_CXX_STANDARD 11)

matlab_add_mex(
    NAME pendulum_propagate_reduced_den
    SRC kernel.cu setup.cu integrate.cu fftSO3R.cu
    LINK_TO cufft cublas
    R2018a
)

set_property(TARGET pendulum_propagate_reduced_den PROPERTY CUDA_ARCHITECTURES 80)

include_directories(~/libcutensor/include)

target_link_libraries(pendulum_propagate_reduced_den ~/libcutensor/lib/11/libcutensor.so)

