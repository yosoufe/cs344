############################################################################
# <summary> CMakeLists.txt for OpenCV and CUDA. </summary>
# <date>    2012-02-07          </date>
# <author>  Quan Tran Minh. edit by Johannes Kast, Michael Sarahan </author>
# <email>   quantm@unist.ac.kr  kast.jo@googlemail.com msarahan@gmail.com</email>
############################################################################
# minimum required cmake version
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
find_package(CUDA QUIET REQUIRED)
#project(HW5)

#set(CUDA_HOST_COMPILER /usr/local/cuda/bin/gcc)
set(CMAKE_BUILD_TYPE Debug)

find_package( OpenCV REQUIRED )


# collect source files

file( GLOB  hdr *.hpp *.h )

SET (HW5_files main.cu student.cu reference_calc.cpp)

CUDA_ADD_EXECUTABLE(HW5 ${HW5_files} ${hdr})
target_link_libraries( HW5 ${OpenCV_LIBS} )


