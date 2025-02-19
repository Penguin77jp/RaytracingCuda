cmake_minimum_required(VERSION 3.10)
project(MyCudaProject)

# C++ 20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX20_STANDARD_COMPILE_OPTION "-std:c++latest")
set(CMAKE_CXX20_EXTENSION_COMPILE_OPTION "-std:c++latest")

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
CUDA_ADD_EXECUTABLE(Main
    sources/main.cu
)

# target_link_libraries(Main HDF5::HDF5)
# target_link_libraries(Main PRIVATE hdf5::hdf5-shared hdf5::hdf5_hl-shared)
target_link_libraries(Main HighFive)
target_link_libraries(Main tinyobjloader::tinyobjloader)
target_link_libraries(Main nlohmann_json::nlohmann_json)
target_link_libraries(Main CLI11::CLI11)



target_include_directories(Main PRIVATE headers)
