cmake_minimum_required(VERSION 3.6)
project(gauss)

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O2")

set(SOURCE_FILES main.cpp linalg.cpp linalg.h linalg.h unique.cpp deb.cpp)
add_executable(gauss ${SOURCE_FILES})
target_link_libraries(gauss ${ARMADILLO_LIBRARIES})
