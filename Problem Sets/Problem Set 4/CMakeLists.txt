############################################################################
# <summary> CMakeLists.txt for OpenCV and CUDA. </summary>
# <date>    2012-02-07          </date>
# <author>  Quan Tran Minh. edit by Johannes Kast, Michael Sarahan </author>
# <email>   quantm@unist.ac.kr  kast.jo@googlemail.com msarahan@gmail.com</email>
############################################################################
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
find_package(CUDA QUIET REQUIRED)
#project(HW4)

#set(CUDA_HOST_COMPILER /usr/local/cuda/bin/gcc)
set(CMAKE_BUILD_TYPE Debug)

find_package( OpenCV REQUIRED )

# collect source files

file( GLOB  hdr *.hpp *.h )
file( GLOB  cu  *.cu)
SET (HW4_files main.cpp loadSaveImage.cpp reference_calc.cpp compare.cpp)

CUDA_ADD_EXECUTABLE(HW4 ${HW4_files} ${hdr} ${img} ${cu})
target_link_libraries( HW4 ${OpenCV_LIBS} )

