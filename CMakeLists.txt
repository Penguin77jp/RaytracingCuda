cmake_minimum_required(VERSION 3.10)
project(MyCudaProject)

# C++ 20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX20_STANDARD_COMPILE_OPTION "-std:c++20")
set(CMAKE_CXX20_EXTENSION_COMPILE_OPTION "-std:c++20")

# CUDA C++ 20
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

# For cli11
find_package(CLI11 CONFIG REQUIRED)

# For nlohmann json
find_package(nlohmann_json CONFIG REQUIRED)

# For obj loader
find_package(tinyobjloader CONFIG REQUIRED)

# For HDF5
# find_package(hdf5 REQUIRED)
# find_package(HDF5 REQUIRED COMPONENTS CXX)
# find_package(hdf5 CONFIG REQUIRED)

# For HighFive
find_package(HighFive REQUIRED)

# For Boost
find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

# For CUDA
set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8")
find_package(CUDA REQUIRED)
message("Cuda version: " ${CUDA_VERSION})

set(SOURCES 
    sources/camera.cpp
    sources/util_json.cpp
)

CUDA_ADD_EXECUTABLE(RenderCuda
    sources/main.cu
    sources/camera.cpp
    sources/util_json.cpp
)

target_link_libraries(RenderCuda HighFive)
target_link_libraries(RenderCuda tinyobjloader::tinyobjloader)
target_link_libraries(RenderCuda nlohmann_json::nlohmann_json)
target_link_libraries(RenderCuda CLI11::CLI11)
target_include_directories(RenderCuda PRIVATE headers)

# lens
CUDA_ADD_EXECUTABLE(Lens_in_Ray
    sources/main_lens_in_ray.cu
    sources/camera.cpp
    sources/util_json.cpp
)

target_link_libraries(Lens_in_Ray HighFive)
target_link_libraries(Lens_in_Ray tinyobjloader::tinyobjloader)
target_link_libraries(Lens_in_Ray nlohmann_json::nlohmann_json)
target_link_libraries(Lens_in_Ray CLI11::CLI11)
target_include_directories(Lens_in_Ray PRIVATE headers)