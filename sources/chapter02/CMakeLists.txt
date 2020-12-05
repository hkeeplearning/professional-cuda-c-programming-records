cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(chapter02 LANGUAGES CXX CUDA)

pccp_module_impl()

add_executable(chapter02
    main.cpp
    sum_arrays.h
    sum_arrays.cu
    sum_arrays.cpp
)

target_link_libraries(chapter02 cudart)

set_property(TARGET chapter02 PROPERTY CUDA_ARCHITECTURES 30 72)