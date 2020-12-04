cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(chapter01 LANGUAGES CXX CUDA)

include_directories(
    ${CMAKE_CUDA_TOOLKIT_INCLUDED_DIRECTORIES}
)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(chapter01
    main.cpp
    hello_world.cu
)

target_link_libraries(chapter01 cudart)