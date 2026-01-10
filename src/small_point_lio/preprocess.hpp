#pragma once
#include "../parameters.hpp"
#include "rose_lm/common.hpp"
#include "voxelgrid_sampling/voxelgrid_sampling.h"
#include <deque>
namespace rose_lm {
namespace small_point_lio {
    class Preprocess {
    public:
        using Ptr = std::unique_ptr<Preprocess>;
        Preprocess(Parameters parameters) {
            params_ = parameters;
        }
        static Ptr create(Parameters p) {
            return std::make_unique<Preprocess>(p);
        }
        void reset() {
            imu_deque.clear();
            point_batch_deque.clear();
            last_lidar_timestamp = -1;
            last_imu_timestamp = -1;
            last_batch_timestamp = -1.0;
        }

        void on_point_cloud_callback(const std::vector<common::Point>& pointcloud) {
            current_batch.points.clear();
            filtered_points.clear();
            filtered_points.reserve(pointcloud.size());
            dense_points.clear();
            dense_points.reserve(pointcloud.size());

            double space_downsample_leaf_size =
                params_.small_point_lio_params.preprocess_params.space_downsample_leaf_size;
            for (size_t i = 0; i < pointcloud.size(); i++) {
                const auto& point = pointcloud[i];
                float dist = point.position.squaredNorm();
                if (dist < params_.small_point_lio_params.preprocess_params.min_distance_squared
                    || dist > params_.small_point_lio_params.preprocess_params.max_distance_squared)
                {
                    continue;
                }
                dense_points.push_back(point);
                if (i % params_.small_point_lio_params.preprocess_params.point_filter_num != 0) {
                    continue;
                }
                if (point.timestamp < last_lidar_timestamp) {
                    continue;
                }
                filtered_points.push_back(point);
            }
            if (space_downsample_leaf_size >= 0.01) {
                downsampler.voxelgrid_sampling_tbb(
                    filtered_points,
                    processed_pointcloud,
                    space_downsample_leaf_size
                );
            } else {
                processed_pointcloud = std::move(filtered_points);
            }

            std::sort(dense_points.begin(), dense_points.end(), [](const auto& x, const auto& y) {
                return x.timestamp < y.timestamp;
            });
            std::sort(
                processed_pointcloud.begin(),
                processed_pointcloud.end(),
                [](const auto& x, const auto& y) { return x.timestamp < y.timestamp; }
            );
            if (!dense_points.empty()) {
                last_timestamp_dense_point = dense_points.back().timestamp;
                dense_point_deque
                    .insert(dense_point_deque.end(), dense_points.begin(), dense_points.end());
            }
            if (!processed_pointcloud.empty()) {
                last_lidar_timestamp = processed_pointcloud.back().timestamp;
                // point_deque.insert(
                //     point_deque.end(),
                //     processed_pointcloud.begin(),
                //     processed_pointcloud.end()
                // );
            }
            for (const auto& p: processed_pointcloud) {
                if (current_batch.points.empty()) {
                    // 第一帧直接开批
                    current_batch.points.push_back(p);
                    last_batch_timestamp = p.timestamp;
                } else if (p.timestamp - last_batch_timestamp < params_.small_point_lio_params.preprocess_params.batch_interval)
                {
                    // 仍属于当前批
                    current_batch.points.push_back(p);
                } else {
                    current_batch.timestamp = current_batch.points.back().timestamp;
                    point_batch_deque.push_back(current_batch);
                    current_batch = common::Batch();
                    current_batch.points.push_back(p);
                    last_batch_timestamp = p.timestamp;
                }
            }
            if (!current_batch.points.empty()) {
                current_batch.timestamp = current_batch.points.back().timestamp;
                point_batch_deque.push_back(current_batch);
                current_batch = common::Batch();
            }
        }
        void on_imu_callback(const common::ImuMsg& imu_msg) {
            if (imu_msg.timestamp < last_imu_timestamp) {
                RCLCPP_ERROR(rclcpp::get_logger("rose_lm"), "imu loop back");
                return;
            }
            imu_deque.emplace_back(imu_msg);
            last_imu_timestamp = imu_msg.timestamp;
        }
        // std::deque<common::Point> point_deque;
        std::deque<common::ImuMsg> imu_deque;
        std::deque<common::Batch> point_batch_deque;
        double last_batch_timestamp = -1.0;
        double last_imu_timestamp = -1.0;
        double last_lidar_timestamp = -1.0;
        voxelgrid_sampling::VoxelgridSampling downsampler;
        std::vector<common::Point> filtered_points;
        std::vector<common::Point> processed_pointcloud;
        common::Batch current_batch;
        std::vector<common::Point> dense_points;
        Parameters params_;

        double last_timestamp_dense_point = -1;
        std::deque<common::Point> dense_point_deque;
    };
} // namespace small_point_lio

} // namespace rose_lm