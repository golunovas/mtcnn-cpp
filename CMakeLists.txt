cmake_minimum_required(VERSION 3.1)
project(mtcnn-cpp)

set (CMAKE_CXX_STANDARD 11)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(Caffe REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${Caffe_INCLUDE_DIRS})

add_executable(mtcnn-cpp face_detector.cpp main.cpp)
target_link_libraries(mtcnn-cpp ${Caffe_LIBRARIES} ${OpenCV_LIBS} )