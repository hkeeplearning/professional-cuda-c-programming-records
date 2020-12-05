macro(pccp_module_impl)
    include_directories(
        ../include
        ${CMAKE_CUDA_TOOLKIT_INCLUDED_DIRECTORIES}
    )

    set(CMAKE_CXX_STANDARD 11)
    set(CMAKE_CXX_STANDARD_REQUIRED True)
endmacro()