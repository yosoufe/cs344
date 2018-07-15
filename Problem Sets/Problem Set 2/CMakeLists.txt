############################################################################
# <summary> CMakeLists.txt for OpenCV and CUDA. </summary>
# <date>    2012-02-07          </date>
# <author>  Quan Tran Minh. edit by Johannes Kast, Michael Sarahan </author>
# <email>   quantm@unist.ac.kr  kast.jo@googlemail.com msarahan@gmail.com</email>
############################################################################

# collect source files

cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(HW2)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "../")

set(CUDA_HOST_COMPILER "/usr/bin/gcc-6")
link_libraries(${OpenCV_LIBS} ${CUDA_LIBRARIES})

# add debugging to CUDA NVCC flags.  For NVidia's NSight tools.
set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} "-G")
set(CMAKE_CUDA_FLAGS "-arch=sm_61")
#set(CUDA_SEPARABLE_COMPILATION ON)
include_directories(${CUDA_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})


file( GLOB  hdr *.hpp *.h )
file( GLOB  cu  *.cu)
set(project_files main.cpp reference_calc.cpp compare.cpp)

CUDA_ADD_EXECUTABLE(${PROJECT_NAME} ${project_files} ${hdr} ${cu})
