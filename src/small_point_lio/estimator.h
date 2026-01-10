/**
 * This file is part of Small Point-LIO, an advanced Point-LIO algorithm implementation.
 * Copyright (C) 2025  Yingjie Huang
 * Licensed under the MIT License. See License.txt in the project root for license information.
 */

#pragma once

#include "eskf.h"
#include "parameters.hpp"
#include "rose_lm/common.hpp"
#include "small_ivox.h"
namespace rose_lm {
namespace small_point_lio {
    class Estimator {
    public:
        using Ptr = std::unique_ptr<Estimator>;
        Estimator(Parameters params);
        static Ptr create(Parameters params) {
            return std::make_unique<Estimator>(params);
        }
        Parameters params_;
        eskf kf;
        // for h_point
        int processed_point_num = 0;
        std::shared_ptr<SmallIVox> ivox;
        Eigen::Matrix<state::value_type, 3, 1> Lidar_T_wrt_IMU;
        Eigen::Matrix<state::value_type, 3, 3> Lidar_R_wrt_IMU;
        Eigen::Vector3f point_lidar_frame;
        Eigen::Vector3f point_odom_frame;
        Eigen::Vector3f point_normal;
        std::vector<Eigen::Vector3f> nearest_points;
        std::vector<Eigen::Vector3f> points_odom_frame;
        common::Batch current_batch;
        std::vector<std::vector<Eigen::Vector3f>> neighbors;
        // std::vector<common::Point> points_lidar_frame;
        // double time_ref = 0;
        // for h_imu
        Eigen::Matrix<state::value_type, 3, 1> angular_velocity;
        Eigen::Matrix<state::value_type, 3, 1> linear_acceleration;
        double imu_acceleration_scale;

        Estimator();

        void reset();

        [[nodiscard]] Eigen::Matrix<state::value_type, state::DIM, state::DIM>
        process_noise_cov() const;

        void h_point(const state& s, point_measurement_result& measurement_result);

        void h_imu(const state& s, imu_measurement_result& measurement_result);
        void h_batch(const state& s, std::vector<point_measurement_result>& results);
    };
} // namespace small_point_lio

} // namespace rose_lm
