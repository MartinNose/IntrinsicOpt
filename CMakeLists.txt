cmake_minimum_required(VERSION 3.0.0)
project(IntrinsicOpt VERSION 0.1.0)

include(CTest)
enable_testing()

find_package(OpenMP)
if (OPENMP_FOUND)
        message(------------------)
        set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
        message(------------------)
endif()

include_directories(/usr/local/Cellar/libomp/14.0.0/include)



list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(Eigen3 REQUIRED)

# LBFGSpp

add_library(lbfgspp INTERFACE)

target_include_directories(lbfgspp INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/utils/LBFGSpp>
  $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/utils/LBFGSpp>
)
target_link_libraries(lbfgspp INTERFACE Eigen3::Eigen)

# MeshTrace

add_library(meshtrace INTERFACE)

target_include_directories(lbfgspp INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/MeshTrace>
  $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include/MeshTrace>
)
target_link_libraries(meshtrace INTERFACE Eigen3::Eigen igl::core)


# Libigl
option(LIBIGL_WITH_OPENGL            "Use OpenGL"         ON)
option(LIBIGL_WITH_OPENGL_GLFW       "Use GLFW"           ON)
include(libigl)
include_directories(utils include) 

set(target_name "IntrinsicOpt")

file(GLOB utils utils/*.h)
file(GLOB include include/*.h)

add_executable(${target_name} main.cpp ${utils} ${include})
set_property(TARGET "IntrinsicOpt" PROPERTY CXX_STANDARD 17)

target_link_libraries(${target_name}
        PUBLIC
        Eigen3::Eigen igl::core igl::opengl_glfw igl::comiso OpenMP::OpenMP_CXX lbfgspp meshtrace
)


set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
