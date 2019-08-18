project(ba_demo)
cmake_minimum_required(VERSION 2.8)
# C++11 support
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMakeModules/)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -Wall")
FIND_PACKAGE(Eigen3 REQUIRED)
FIND_PACKAGE(CSparse REQUIRED)
FIND_PACKAGE(Cholmod REQUIRED)
FIND_PACKAGE(G2O REQUIRED)
FIND_PACKAGE(PCL REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(cvsba REQUIRED)
FIND_PACKAGE(DBoW2 REQUIRED)
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${EIGEN3_INCLUDE_DIRS}
  ${CSPARSE_INCLUDE_DIR}
  ${Cholmod_INCLUDE_DIR}
  ${G2O_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${DBoW2_INCLUDE_DIR}
  /usr/include/suitesparse
)
LIST(APPEND G2O_LIBS
  cxsparse
  cholmod
  g2o_cli g2o_ext_freeglut_minimal g2o_simulator
  g2o_solver_slam2d_linear g2o_types_icp g2o_types_slam2d
  g2o_core g2o_interface g2o_solver_csparse g2o_solver_structure_only
  g2o_types_sba g2o_types_slam3d g2o_csparse_extension
  g2o_opengl_helper g2o_solver_dense g2o_stuff
  g2o_types_sclam2d g2o_parser g2o_solver_pcg
  g2o_types_data g2o_types_sim3
)
link_directories(${PCL_LIBRARY_DIRS})
link_directories(${OpenCV_LIB_DIR})
add_definitions(${PCL_DEFINITIONS})
aux_source_directory(. DIR_SRCS)
add_executable(ba_demo ${DIR_SRCS})
target_link_libraries(ba_demo
${G2O_LIBS}
${PCL_LIBRARIES}
${OpenCV_LIBS}
${cvsba_LIBS}
${DBoW2_LIBRARY}
)


