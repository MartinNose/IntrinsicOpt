add_library(point_sample point_sample.cpp)

target_link_libraries(point_sample
        PRIVATE
        Eigen3::Eigen igl::core
    )