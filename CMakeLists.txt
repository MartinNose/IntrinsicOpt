cmake_minimum_required(VERSION 3.0.0)
project(IntrinsicOpt VERSION 0.1.0)

include(CTest)
enable_testing()

list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(Eigen3 REQUIRED)

# Libigl
option(LIBIGL_WITH_OPENGL            "Use OpenGL"         ON)
option(LIBIGL_WITH_OPENGL_GLFW       "Use GLFW"           ON)
include(libigl)
include_directories(utils point_sample)

set(target_name "IntrinsicOpt")
file(GLOB utils utils/*.h)

add_subdirectory(point_sample)

add_executable(${target_name} main.cpp ${utils})

target_link_libraries(${target_name}
        PUBLIC
        Eigen3::Eigen igl::core igl::opengl_glfw igl::comiso
        point_sample
        )

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
