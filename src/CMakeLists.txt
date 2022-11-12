cmake_minimum_required(VERSION 3.10)

set(SOURCE_FILES
  ${PROJECT_SOURCE_DIR}/src/image_registration.cpp
  ${PROJECT_SOURCE_DIR}/src/SuperGlueMatcher.cpp
)

add_library(${LIBRARY_NAME}
  SHARED
    ${SOURCE_FILES}
)

target_include_directories(${LIBRARY_NAME}
  SYSTEM PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${OpenCV_INCLUDE_DIRS}
  PRIVATE
)

target_link_libraries(${LIBRARY_NAME}
  PUBLIC
    ${OpenCV_LIBRARIES}
  PRIVATE
    torch_cpp
)

target_compile_options(${LIBRARY_NAME}
  PRIVATE
     $<$<CONFIG:Debug>:-O0 -g -Wall -Werror>
     $<$<CONFIG:Release>:-O3 -Wall -Werror>
)

set_target_properties(${LIBRARY_NAME}
  PROPERTIES
    $<$<COMPILE_LANGUAGE:CUDA>:CUDA_STANDARD 14>
)

target_compile_features(${LIBRARY_NAME}
  PRIVATE
    cxx_std_17
)
