cmake_minimum_required(VERSION 4.2)
project(TP1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

# Ne pas compiler les exemples et tests de la bibliothèque.
set(MATPLOTPP_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(MATPLOTPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
        matplotplusplus
        GIT_REPOSITORY https://github.com/alandefreitas/matplotplusplus.git
        GIT_TAG v1.2.2
        GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(matplotplusplus)

if (MSVC)
    target_compile_options(matplot PRIVATE /wd4834)
endif ()

find_package(Threads REQUIRED)

add_executable(TP1 main.cpp)

target_link_libraries(TP1
        PRIVATE
        matplot
        Threads::Threads
)

# CUDA obligatoire pour compiler et tester les kernels.
find_package(CUDAToolkit REQUIRED)
enable_language(CUDA)

target_sources(TP1 PRIVATE gpu.cu)

target_link_libraries(TP1 PRIVATE CUDA::cudart)

target_compile_definitions(TP1 PRIVATE
        HAS_CUDA=1
        HAS_CUDA_KERNELS=1
)

set_target_properties(TP1 PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_EXTENSIONS OFF
        CUDA_ARCHITECTURES 89
)

message(STATUS "CUDA Toolkit : ${CUDAToolkit_VERSION}")
message(STATUS "Compilateur CUDA : ${CMAKE_CUDA_COMPILER}")