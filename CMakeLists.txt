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

find_package(Threads REQUIRED)

add_executable(TP1 main.cpp)

target_link_libraries(TP1
        PRIVATE
        matplot
        Threads::Threads
)

# Support CUDA optionnel.
find_package(CUDAToolkit QUIET)

if(CUDAToolkit_FOUND)
    target_link_libraries(TP1 PRIVATE CUDA::cudart)
    target_compile_definitions(TP1 PRIVATE HAS_CUDA=1)

    message(STATUS "CUDA Toolkit detecte : ${CUDAToolkit_VERSION}")

    # Spécificités du compilateur CUDA
    include(CheckLanguage)
    check_language(CUDA)

    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)

        set_target_properties(TP1 PROPERTIES
                CUDA_STANDARD 17
                CUDA_STANDARD_REQUIRED ON
        )

        message(STATUS "Compilateur CUDA disponible : ${CMAKE_CUDA_COMPILER}")
    else()
        message(STATUS "Compilation des fichiers GPU indisponible")
    endif()
else()
    message(STATUS "CUDA Toolkit absent : compilation CPU")
endif()