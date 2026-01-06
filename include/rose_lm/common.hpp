#pragma once
#include <Eigen/Dense>

namespace rose_lm {
namespace common {
    struct Odometry {
        double timestamp; // Unit: s
        Eigen::Vector3d position; // Unit: m
        Eigen::Vector3d velocity; // Unit: m/s
        Eigen::Quaterniond orientation; // Unit: rad
        Eigen::Vector3d angular_velocity; // Unit: rad/s
    };

    struct ImuMsg {
        double timestamp; // Unit: s
        Eigen::Vector3d linear_acceleration; // Unit: g
        Eigen::Vector3d angular_velocity; // Unit: rad/s
    };

    struct Point {
        double timestamp; // Unit: s
        Eigen::Vector3f position; // Unit: m
        int count = 1;
    };
    struct Batch {
        double timestamp; // Unit: s
        std::vector<Point> points; // Unit: m
    };
} // namespace common

} // namespace rose_lm