cmake_minimum_required(VERSION 3.10)

add_executable(super_glue_matcher_app
  ${CMAKE_CURRENT_LIST_DIR}/SuperGlueMatcherApp.cpp
)

target_link_libraries(super_glue_matcher_app
  PUBLIC
    ${LIBRARY_NAME}
)

# ------------------------------------------------------------------------------

add_executable(super_glue_matcher_micasense_app
  ${CMAKE_CURRENT_LIST_DIR}/SuperGlueMatcherMicasenseApp.cpp
)

target_link_libraries(super_glue_matcher_micasense_app
  PUBLIC
    ${LIBRARY_NAME}
)

# ------------------------------------------------------------------------------

add_executable(super_glue_matcher_two_rgbs
  ${CMAKE_CURRENT_LIST_DIR}/SuperGlueMatcherTwoRGBs.cpp
)

target_link_libraries(super_glue_matcher_two_rgbs
  PUBLIC
    ${LIBRARY_NAME}
)
