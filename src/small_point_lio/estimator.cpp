/**
 * This file is part of Small Point-LIO, an advanced Point-LIO algorithm implementation.
 * Copyright (C) 2025  Yingjie Huang
 * Licensed under the MIT License. See License.txt in the project root for license information.
 */

#include "estimator.h"
#include "utils.hpp"
#include <tbb/tbb.h>
namespace rose_lm {
namespace small_point_lio {
    constexpr int NUM_MATCH_POINTS = 5;

    Estimator::Estimator(Parameters params) {
        params_ = params;
        Lidar_R_wrt_IMU =
            params_.small_point_lio_params.estimator_params.extrinsic_R.cast<state::value_type>();
        Lidar_T_wrt_IMU =
            params_.small_point_lio_params.estimator_params.extrinsic_T.cast<state::value_type>();
        if (params_.small_point_lio_params.estimator_params.extrinsic_est_en) {
            kf.x.offset_T_L_I =
                params_.small_point_lio_params.estimator_params.extrinsic_T.cast<state::value_type>(
                );
            kf.x.offset_R_L_I =
                params_.small_point_lio_params.estimator_params.extrinsic_R.cast<state::value_type>(
                );
        }
        imu_acceleration_scale = params_.small_point_lio_params.estimator_params.gravity.norm()
            / params_.small_point_lio_params.estimator_params.acc_norm;
        kf.max_iter = params_.small_point_lio_params.estimator_params.max_iter;
        kf.init(
            [this](auto&& s, auto&& measurement_result) { return h_point(s, measurement_result); },
            [this](auto&& s, auto&& measurement_result) { return h_imu(s, measurement_result); },
            [this](auto&& s, auto&& measurement_result) { return h_batch(s, measurement_result); }
        );
    }

    void Estimator::reset() {
        ivox = std::make_shared<SmallIVox>(
            params_.small_point_lio_params.estimator_params.map_resolution,
            1000000
        );
        kf.P = Eigen::Matrix<state::value_type, state::DIM, state::DIM>::Identity() * 0.01;
        kf.P.block<3, 3>(state::gravity_index, state::gravity_index).diagonal().fill(0.0001);
        kf.P.block<3, 3>(state::bg_index, state::bg_index).diagonal().fill(0.001);
        kf.P.block<3, 3>(state::ba_index, state::ba_index).diagonal().fill(0.001);
    }

    [[nodiscard]] Eigen::Matrix<state::value_type, state::DIM, state::DIM>
    Estimator::process_noise_cov() const {
        Eigen::Matrix<state::value_type, state::DIM, state::DIM> cov =
            Eigen::Matrix<state::value_type, state::DIM, state::DIM>::Zero();
        cov.block<3, 3>(state::velocity_index, state::velocity_index)
            .diagonal()
            .fill(static_cast<state::value_type>(
                params_.small_point_lio_params.estimator_params.velocity_cov
            ));
        cov.block<3, 3>(state::omg_index, state::omg_index)
            .diagonal()
            .fill(static_cast<state::value_type>(
                params_.small_point_lio_params.estimator_params.omg_cov
            ));
        cov.block<3, 3>(state::acceleration_index, state::acceleration_index)
            .diagonal()
            .fill(static_cast<state::value_type>(
                params_.small_point_lio_params.estimator_params.acceleration_cov
            ));
        cov.block<3, 3>(state::bg_index, state::bg_index)
            .diagonal()
            .fill(static_cast<state::value_type>(
                params_.small_point_lio_params.estimator_params.bg_cov
            ));
        cov.block<3, 3>(state::ba_index, state::ba_index)
            .diagonal()
            .fill(static_cast<state::value_type>(
                params_.small_point_lio_params.estimator_params.ba_cov
            ));
        return cov;
    }
    static inline Eigen::Matrix3d hat(const Eigen::Vector3d& v) {
        Eigen::Matrix3d m;
        m << 0.0, -v.z(), v.y(), v.z(), 0.0, -v.x(), -v.y(), v.x(), 0.0;
        return m;
    }
    void Estimator::h_batch(const state& s, std::vector<point_measurement_result>& results) {
        results.clear();
        const size_t N = current_batch.points.size();
        results.reserve(N / 2);

        // --- prefetch constants ---
        const bool ext_on = params_.small_point_lio_params.estimator_params.extrinsic_est_en;
        const Eigen::Matrix3d R_LI_d = ext_on ? kf.x.offset_R_L_I : Lidar_R_wrt_IMU.cast<double>();
        const Eigen::Vector3d T_LI_d = ext_on ? kf.x.offset_T_L_I : Lidar_T_wrt_IMU.cast<double>();
        const double wnorm = kf.x.omg.norm();
        const double plane_thr = params_.small_point_lio_params.estimator_params.plane_threshold;
        const double match_s = params_.small_point_lio_params.estimator_params.match_sqaured;
        const double laser_cov = params_.small_point_lio_params.estimator_params.laser_point_cov;
        const double curv_thr = params_.small_point_lio_params.estimator_params.curv_threshold;
        const double laser_distance_cov_ratio =
            params_.small_point_lio_params.estimator_params.laser_distance_cov_ratio;
        // Use float for point storage to reduce memory bandwidth (tune to your needs)
        points_odom_frame.clear();
        points_odom_frame.resize(N, Eigen::Vector3f::Zero());

        // pts_imu as float to match odom frame. Compute in double then cast once.
        std::vector<Eigen::Vector3f> pts_imu_f;
        pts_imu_f.resize(N);

        // Precompute transformed lidar points in IMU frame (pts_imu_f) and odom frame
        const Eigen::Matrix3d& R_LI = R_LI_d;
        const Eigen::Vector3d& T_LI = T_LI_d;
        const auto kf_rot = kf.x.rotation; // assume double matrix
        const auto kf_pos = kf.x.position;
        const auto kf_vel = kf.x.velocity;

        for (size_t i = 0; i < N; ++i) {
            const auto& point_lidar = current_batch.points[i];
            const double dt = point_lidar.timestamp - current_batch.timestamp;

            // pts_imu in double then cast to float once
            const Eigen::Vector3d pt_imu_d = R_LI * point_lidar.position.cast<double>() + T_LI;
            pts_imu_f[i] = pt_imu_d.cast<float>();

            // compute small rotation R_delta (first-order approx) in double
            Eigen::Matrix3d R_delta = Eigen::Matrix3d::Identity();
            if (wnorm > 1e-8)
                R_delta += hat(kf.x.omg * dt);

            // combine rotation once and cast result to float
            const Eigen::Vector3d pt_in_odom_d =
                (kf_rot * R_delta * pt_imu_d + kf_pos + kf_vel * dt);
            points_odom_frame[i] = pt_in_odom_d.cast<float>();
        }
        {
            // static double total_time = 0.0;
            // auto start = std::chrono::steady_clock::now();
            // //ivox->get_closest_points_batch(points_odom_frame, neighbors, NUM_MATCH_POINTS);
            // auto end = std::chrono::steady_clock::now();
            // total_time += std::chrono::duration<double>(end - start).count();
            // utils::XSecOnce(
            //     [&]() {
            //         std::cout << "ivox get_closest_points_batch time: " << total_time * 1000.0
            //                   << " ms" << std::endl;
            //         total_time = 0.0;
            //     },
            //     1.0
            // );
        }

        {
            static double total_time = 0.0;
            auto start = std::chrono::steady_clock::now();
            // Per-thread result vectors to avoid high-overhead concurrent push
            tbb::enumerable_thread_specific<std::vector<point_measurement_result>> local_results(
                []() { return std::vector<point_measurement_result>(); }
            );

            // Per-thread eigen solver to avoid repeated construction
            tbb::enumerable_thread_specific<Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>>
                local_solver([]() { return Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>(); });

            // Parallel loop
            tbb::parallel_for(
                tbb::blocked_range<size_t>(0, N),
                [&](const tbb::blocked_range<size_t>& r) {
                    auto& thread_res = local_results.local();
                    auto& solver = local_solver.local();

                    for (size_t i = r.begin(); i != r.end(); ++i) {
                        std::vector<Eigen::Vector3f> near;
                        ivox->get_closest_point(points_odom_frame[i], near, NUM_MATCH_POINTS);
                        //const auto& near = neighbors[i];
                        if (near.size() < NUM_MATCH_POINTS)
                            continue;

                        Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
                        for (const auto& np: near) {
                            const Eigen::Vector3d p = np.cast<double>();
                            centroid += p;
                        }
                        centroid /= static_cast<double>(near.size());

                        // Compute covariance (3x3) in double - one pass
                        Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
                        for (const auto& np: near) {
                            const Eigen::Vector3d d = np.cast<double>() - centroid;
                            cov.noalias() += d * d.transpose();
                        }
                        if (near.size() > 1)
                            cov /= static_cast<double>(near.size() - 1);
                        else
                            continue;

                        // Per-thread solver reuse
                        solver.compute(cov);
                        if (solver.info() != Eigen::Success)
                            continue;

                        const Eigen::Vector3d n =
                            solver.eigenvectors().col(0); // smallest eigenvector
                        const double lambda0 = solver.eigenvalues()[0];
                        const double lambda_sum = solver.eigenvalues().sum();
                        const double curv = lambda0 / (lambda_sum + 1e-9);
                        if (curv > curv_thr)
                            continue;

                        const double d_plane = -n.dot(centroid);

                        // quick residual check using float odom point (cheap)
                        const Eigen::Vector3f pt_odom_f = points_odom_frame[i];
                        const double d_signed = n.dot(pt_odom_f.cast<double>()) + d_plane;

                        // match_s check: original used pt_lidar.norm() <= match_s * d_signed^2
                        const double pt_lidar_norm = current_batch.points[i].position.norm();
                        if (pt_lidar_norm <= match_s * d_signed * d_signed)
                            continue;

                        // fine-grained point-to-plane check for neighbors (full double)
                        bool valid = true;
                        for (const auto& np: near) {
                            if (std::abs(n.dot(np.cast<double>()) + d_plane) > plane_thr) {
                                valid = false;
                                break;
                            }
                        }
                        if (!valid)
                            continue;

                        // Build measurement result
                        point_measurement_result mr {};
                        mr.valid = true;
                        mr.laser_point_cov = laser_cov
                            * log(current_batch.points[i].position.norm() + 1)
                            * laser_distance_cov_ratio;
                        mr.z = -d_signed;

                        // Build H
                        if (ext_on) {
                            Eigen::Matrix<state::value_type, 3, 1> normal0 =
                                n.cast<state::value_type>();
                            Eigen::Matrix<state::value_type, 3, 1> C;
                            C.noalias() = s.rotation.transpose() * normal0;
                            Eigen::Matrix<state::value_type, 3, 1> A, B;
                            A.noalias() = pts_imu_f[i].cast<state::value_type>().cross(C);
                            B.noalias() = point_lidar_frame.cast<state::value_type>().cross(
                                s.offset_R_L_I.transpose() * C
                            );
                            mr.H << normal0.transpose(), A.transpose(), B.transpose(),
                                C.transpose();
                        } else {
                            Eigen::Matrix<state::value_type, 3, 1> normal0 =
                                n.cast<state::value_type>();
                            Eigen::Matrix<state::value_type, 3, 1> A;
                            A.noalias() = pts_imu_f[i].cast<state::value_type>().cross(
                                s.rotation.transpose() * normal0
                            );
                            // fill rest with zeros (explicit)
                            mr.H << normal0.transpose(), A.transpose(), 0.0, 0.0, 0.0, 0.0, 0.0,
                                0.0;
                        }

                        if (current_batch.points[i].count > 1)
                            mr.count = current_batch.points[i].count;

                        thread_res.emplace_back(std::move(mr));
                    }
                }
            );

            // Merge thread-local results
            size_t total = 0;
            for (auto& v: local_results)
                total += v.size();
            results.reserve(total);
            for (auto& v: local_results) {
                if (!v.empty()) {
                    results.insert(
                        results.end(),
                        std::make_move_iterator(v.begin()),
                        std::make_move_iterator(v.end())
                    );
                }
            }
            auto end = std::chrono::steady_clock::now();
            total_time += std::chrono::duration<double>(end - start).count();
            utils::XSecOnce(
                [&]() {
                    std::cout << "tbb_plane_batch time: " << total_time * 1000.0 << " ms"
                              << std::endl;
                    total_time = 0.0;
                },
                1.0
            );
        }
    }
    void Estimator::h_point(const state& s, point_measurement_result& measurement_result) {
        measurement_result.valid = false;
        // get closest point
        Eigen::Matrix<state::value_type, 3, 1> point_imu_frame;
        if (params_.small_point_lio_params.estimator_params.extrinsic_est_en) {
            point_imu_frame =
                kf.x.offset_R_L_I * point_lidar_frame.cast<state::value_type>() + kf.x.offset_T_L_I;
        } else {
            point_imu_frame =
                Lidar_R_wrt_IMU * point_lidar_frame.cast<state::value_type>() + Lidar_T_wrt_IMU;
        }
        point_odom_frame = (kf.x.rotation * point_imu_frame + kf.x.position).cast<float>();
        ivox->get_closest_point(point_odom_frame, nearest_points, NUM_MATCH_POINTS);
        if (nearest_points.size() != NUM_MATCH_POINTS) {
            return;
        }
        // estimate plane
#if 0
        Eigen::Matrix<float, NUM_MATCH_POINTS, 3> A;
        for (int j = 0; j < NUM_MATCH_POINTS; j++) {
            A.row(j) = nearest_points[j];
        }
        Eigen::Matrix<float, NUM_MATCH_POINTS, 1> b;
        b.setConstant(-1);
        Eigen::Vector3f normal = A.colPivHouseholderQr().solve(b);
        float d = 1.0f / normal.norm();
#else
        Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
        for (const auto& p: nearest_points) {
            centroid.noalias() += p;
        }
        centroid /= static_cast<float>(nearest_points.size());
        Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
        for (const auto& p: nearest_points) {
            Eigen::Vector3f centered = p - centroid;
            covariance.noalias() += centered * centered.transpose();
        }
        covariance /= static_cast<float>(nearest_points.size() - 1);
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);
        Eigen::Vector3f normal = solver.eigenvectors().col(0);
        point_normal = normal.normalized();
        float d = -normal.dot(centroid);
#endif
        for (int j = 0; j < NUM_MATCH_POINTS; j++) {
            float point_distanace = std::abs(normal.dot(nearest_points[j]) + d);
            if (point_distanace > params_.small_point_lio_params.estimator_params.plane_threshold) {
                return;
            }
        }
        float point_distanace = normal.dot(point_odom_frame) + d;
        if (point_lidar_frame.norm()
            <= params_.small_point_lio_params.estimator_params.match_sqaured * point_distanace
                * point_distanace)
        {
            return;
        }
        // calculate residual and jacobian matrix
        measurement_result.laser_point_cov = static_cast<state::value_type>(
            params_.small_point_lio_params.estimator_params.laser_point_cov
        );
        if (params_.small_point_lio_params.estimator_params.extrinsic_est_en) {
            Eigen::Matrix<state::value_type, 3, 1> normal0 = normal.cast<state::value_type>();
            Eigen::Matrix<state::value_type, 3, 1> C = s.rotation.transpose() * normal0;
            Eigen::Matrix<state::value_type, 3, 1> A, B;
            A.noalias() = point_imu_frame.cross(C);
            B.noalias() =
                point_lidar_frame.cast<state::value_type>().cross(s.offset_R_L_I.transpose() * C);
            measurement_result.H << normal0.transpose(), A.transpose(), B.transpose(),
                C.transpose();
        } else {
            Eigen::Matrix<state::value_type, 3, 1> normal0 = normal.cast<state::value_type>();
            Eigen::Matrix<state::value_type, 3, 1> A;
            A.noalias() = point_imu_frame.cross(s.rotation.transpose() * normal0);
            measurement_result.H << normal0.transpose(), A.transpose(), 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0;
        }
        measurement_result.z = -point_distanace;
        measurement_result.valid = true;
    }

    void Estimator::h_imu(const state& s, imu_measurement_result& measurement_result) {
        std::memset(measurement_result.satu_check, false, 6);
        measurement_result.z.segment<3>(0) = angular_velocity - s.omg - s.bg;
        measurement_result.z.segment<3>(3) =
            linear_acceleration * imu_acceleration_scale - s.acceleration - s.ba;
        measurement_result.imu_meas_omg_cov = static_cast<state::value_type>(
            params_.small_point_lio_params.estimator_params.imu_meas_omg_cov
        );
        measurement_result.imu_meas_acc_cov = static_cast<state::value_type>(
            params_.small_point_lio_params.estimator_params.imu_meas_acc_cov
        );
        if (params_.small_point_lio_params.estimator_params.check_satu) {
            for (int i = 0; i < 3; i++) {
                if (std::abs(angular_velocity(i))
                    >= params_.small_point_lio_params.estimator_params.satu_gyro) {
                    measurement_result.satu_check[i] = true;
                    measurement_result.z(i) = 0.0;
                }
                if (std::abs(linear_acceleration(i))
                    >= params_.small_point_lio_params.estimator_params.satu_acc) {
                    measurement_result.satu_check[i + 3] = true;
                    measurement_result.z(i + 3) = 0.0;
                }
            }
        }
    }
} // namespace small_point_lio

} // namespace rose_lm
