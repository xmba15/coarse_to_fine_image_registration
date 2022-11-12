cmake_minimum_required(VERSION 3.10)

project(image_registration LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

set(LIBRARY_NAME ${PROJECT_NAME})

find_package(OpenCV REQUIRED)
find_package(torch_cpp REQUIRED)

add_subdirectory(src)

if(BUILD_EXAMPLES)
  add_subdirectory(examples)
endif(BUILD_EXAMPLES)
