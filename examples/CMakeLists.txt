cmake_minimum_required(VERSION 3.10)

add_executable(super_glue_matcher_app
  ${CMAKE_CURRENT_LIST_DIR}/SuperGlueMatcherApp.cpp
)

target_link_libraries(super_glue_matcher_app
  PUBLIC
    ${LIBRARY_NAME}
)
