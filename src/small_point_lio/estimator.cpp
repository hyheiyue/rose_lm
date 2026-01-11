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
    constexpr int MIN_MATCH_POINTS = 5;
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
            params,
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
    std::vector<point_measurement_result> all_results;

    void Estimator::h_batch(const state& s, std::vector<point_measurement_result>& results) {
        auto last_results = results;
        results.clear();
        const size_t N = current_batch.points.size();
        results.reserve(N);

        const bool ext_on = params_.small_point_lio_params.estimator_params.extrinsic_est_en;
        const Eigen::Matrix3d R_LI_d = ext_on ? s.offset_R_L_I : Lidar_R_wrt_IMU.cast<double>();
        const Eigen::Vector3d T_LI_d = ext_on ? s.offset_T_L_I : Lidar_T_wrt_IMU.cast<double>();
        const double wnorm = s.omg.norm();
        const double plane_thr = params_.small_point_lio_params.estimator_params.plane_threshold;
        const double match_s = params_.small_point_lio_params.estimator_params.match_sqaured;
        const double laser_cov = params_.small_point_lio_params.estimator_params.laser_point_cov;
        // Use float for point storage to reduce memory bandwidth (tune to your needs)

        if (s.batch_iter == 0) {
            points_odom_frame.clear();
            points_odom_frame.resize(N, Eigen::Vector3f::Zero());
            valid_ids.clear();
            valid_ids.reserve(N);
            point_converged.clear();
            point_converged.resize(N, 0);
            is_valid.clear();
            is_valid.resize(N, 0);
            all_results.clear();
            all_results.resize(N, {});
        }
        // pts_imu as float to match odom frame. Compute in double then cast once.
        std::vector<Eigen::Vector3f> pts_imu_f;
        pts_imu_f.resize(N);

        // Precompute transformed lidar points in IMU frame (pts_imu_f) and odom frame
        const Eigen::Matrix3d& R_LI = R_LI_d;
        const Eigen::Vector3d& T_LI = T_LI_d;
        const auto kf_rot = s.rotation; // assume double matrix
        const auto kf_pos = s.position;
        const auto kf_vel = s.velocity;
        const Eigen::Vector3d w = s.omg;
        auto hat = [](const Eigen::Vector3d& v) {
            return (Eigen::Matrix3d() << 0, -v.z(), v.y(), v.z(), 0, -v.x(), -v.y(), v.x(), 0)
                .finished();
        };
        if (s.batch_iter == 1) {
            for (size_t id: valid_ids)
                is_valid[id] = 1;

            for (size_t i = 0; i < N; ++i) {
                if (!is_valid[i]) {
                    ivox->add_point(points_odom_frame[i]);
                }
            }
        }

        const float converged_thresh_sq =
            params_.small_point_lio_params.estimator_params.map_resolution / 2.0
            * params_.small_point_lio_params.estimator_params.map_resolution / 2.0;
        for (size_t i = 0; i < N; ++i) {
            const auto& point_lidar = current_batch.points[i];
            const double dt = point_lidar.timestamp - current_batch.timestamp;
            // pts_imu in double then cast to float once
            const Eigen::Vector3d pt_imu_d = R_LI * point_lidar.position.cast<double>() + T_LI;
            pts_imu_f[i] = pt_imu_d.cast<float>();
            // compute small rotation R_delta (first-order approx) in double
            Eigen::Matrix3d R_delta = Eigen::Matrix3d::Identity();
            if (wnorm > 1e-8)
                R_delta += hat(w / wnorm * (wnorm * dt));
            // combine rotation once and cast result to float
            const Eigen::Vector3d pt_in_odom_d =
                ((kf_rot * R_delta) * pt_imu_d + kf_pos + kf_vel * dt);
            Eigen::Vector3f pt_odom_f = pt_in_odom_d.cast<float>();
            if (s.batch_iter != 0 && point_converged[i] != 1) {
                //const Eigen::Vector3f diff = pt_odom_f - points_odom_frame[i];
                auto k_cur = ivox->get_position_index(pt_odom_f);
                auto k_last = ivox->get_position_index(points_odom_frame[i]);
                if (k_cur == k_last) {
                    point_converged[i] = 1;
                    ivox->add_point(points_odom_frame[i]);
                    if (is_valid[i])
                        converged_count++;
                } else {
                    point_converged[i] = 0;
                }
            }
            points_odom_frame[i] = pt_odom_f;
        }

        {
            // Per-thread result vectors to avoid high-overhead concurrent push
            // Thread-local containers

            tbb::enumerable_thread_specific<Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>>
                local_solver([]() { return Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>(); });
            if (all_ids.size() != N) {
                all_ids.resize(N);
                std::iota(all_ids.begin(), all_ids.end(), 0);
            }
            const std::vector<size_t>& ids_to_process = (s.batch_iter == 0) ? all_ids : valid_ids;
            tbb::parallel_for(
                tbb::blocked_range<size_t>(0, ids_to_process.size(), ids_to_process.size() / 5),
                [&](const tbb::blocked_range<size_t>& r) {
                    auto& solver = local_solver.local();

                    for (size_t k = r.begin(); k != r.end(); ++k) {
                        const size_t i = ids_to_process[k];
                        if (point_converged[i] == 1) {
                            continue;
                        }
                        std::vector<Eigen::Vector3f> near;
                        ivox->get_closest_point(points_odom_frame[i], near, NUM_MATCH_POINTS);

                        if (near.size() < MIN_MATCH_POINTS)
                            continue;
                        Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
                        for (const auto& np: near)
                            centroid += np.cast<double>();
                        centroid /= static_cast<double>(near.size());

                        Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
                        for (const auto& np: near) {
                            Eigen::Vector3d d = np.cast<double>() - centroid;
                            cov.noalias() += d * d.transpose();
                        }

                        if (near.size() > 1)
                            cov /= static_cast<double>(near.size() - 1);
                        else
                            continue;

                        solver.compute(cov);
                        const Eigen::Vector3d n = solver.eigenvectors().col(0);

                        const double d_plane = -n.dot(centroid);

                        const Eigen::Vector3d pt_odom_d = points_odom_frame[i].cast<double>();
                        const double d_signed = n.dot(pt_odom_d) + d_plane;
                        if (s.batch_iter == 0) {
                            const double pt_norm = current_batch.points[i].position.norm();
                            if (pt_norm <= match_s * d_signed * d_signed)
                                continue;
                            bool valid = true;
                            for (const auto& np: near) {
                                if (std::abs(n.dot(np.cast<double>()) + d_plane) > plane_thr) {
                                    valid = false;
                                    break;
                                }
                            }
                            if (!valid)
                                continue;
                        }
                        point_measurement_result mr {};
                        mr.valid = true;
                        mr.z = -d_signed;
                        mr.laser_point_cov = laser_cov;
                        mr.count = current_batch.points[i].count;
                        mr.id = i;

                        const Eigen::Matrix<state::value_type, 3, 1> normal0 =
                            n.cast<state::value_type>();
                        if (ext_on) {
                            const Eigen::Matrix<state::value_type, 3, 1> C =
                                s.rotation.transpose() * normal0;

                            const Eigen::Matrix<state::value_type, 3, 1> A =
                                pts_imu_f[i].cast<state::value_type>().cross(C);

                            const Eigen::Matrix<state::value_type, 3, 1> B =
                                point_lidar_frame.cast<state::value_type>().cross(
                                    s.offset_R_L_I.transpose() * C
                                );
                            mr.H << normal0.transpose(), A.transpose(), B.transpose(),
                                C.transpose();
                        } else {
                            const Eigen::Matrix<state::value_type, 3, 1> A =
                                pts_imu_f[i].cast<state::value_type>().cross(
                                    s.rotation.transpose() * normal0
                                );
                            mr.H << normal0.transpose(), A.transpose(), 0.0, 0.0, 0.0, 0.0, 0.0,
                                0.0;
                        }
                        all_results[i] = mr;
                    }
                }
            );
            results.clear();
            results = all_results;
            if (s.batch_iter == 0) {
                for (size_t i = 0; i < results.size(); i++) {
                    if (results[i].valid) {
                        valid_ids.push_back(results[i].id);
                    }
                }
            }

            processed_point_num = valid_ids.size();
        }
    }
    void Estimator::h_point(const state& s, point_measurement_result& measurement_result) {
        measurement_result.valid = false;
        // get closest point
        Eigen::Matrix<state::value_type, 3, 1> point_imu_frame;
        if (params_.small_point_lio_params.estimator_params.extrinsic_est_en) {
            point_imu_frame =
                s.offset_R_L_I * point_lidar_frame.cast<state::value_type>() + s.offset_T_L_I;
        } else {
            point_imu_frame =
                Lidar_R_wrt_IMU * point_lidar_frame.cast<state::value_type>() + Lidar_T_wrt_IMU;
        }
        point_odom_frame = (s.rotation * point_imu_frame + s.position).cast<float>();
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
