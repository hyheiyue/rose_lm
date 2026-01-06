#pragma once
#include <Eigen/Core>
#include <rclcpp/node.hpp>
namespace rose_lm {
class Parameters {
public:
    Parameters() {}
    void load(rclcpp::Node& node) {
        small_point_lio_params.load(node);
    }
    struct SmallPointLioParams {
        struct PreprocessParams {
            double min_distance_squared;
            double max_distance_squared;
            double space_downsample_leaf_size = 0.1;
            double batch_interval = 0.01;
            int point_filter_num = 1;
            void load(rclcpp::Node& node) {
                double min_distance;
                min_distance = node.declare_parameter<double>("small_point_lio.min_distance", 0.05);
                min_distance_squared = min_distance * min_distance;
                double max_distance;
                max_distance =
                    node.declare_parameter<double>("small_point_lio.max_distance", 100.0);
                max_distance_squared = max_distance * max_distance;
                space_downsample_leaf_size = node.declare_parameter<double>(
                    "small_point_lio.space_downsample_leaf_size",
                    space_downsample_leaf_size
                );
                batch_interval = node.declare_parameter<double>(
                    "small_point_lio.batch_interval",
                    batch_interval
                );
                point_filter_num = node.declare_parameter<int>(
                    "small_point_lio.point_filter_num",
                    point_filter_num
                );
            }
        } preprocess_params;
        struct EstimatorParams {
            double map_resolution = 0.1;
            bool extrinsic_est_en = false;
            double laser_point_cov = 0.01;
            double laser_distance_cov_ratio = 0.1;
            double imu_meas_acc_cov = 0.01;
            double imu_meas_omg_cov = 0.01;

            double velocity_cov = 20.0;
            double acceleration_cov = 500.0;
            double omg_cov = 1000.0;
            double ba_cov = 0.0001;
            double bg_cov = 0.0001;
            double plane_threshold = 0.1;
            double match_sqaured = 81.0;
            double curv_threshold = 0.1;
            bool check_satu = true;
            double satu_acc = 3.0;
            double satu_gyro = 35.0;
            double acc_norm = 1.0;

            Eigen::Vector3d extrinsic_T;
            Eigen::Matrix3d extrinsic_R;
            Eigen::Vector3d gravity;
            int init_map_size = 10;
            bool fix_gravity_direction = true;
            int max_iter = 5;
            void load(rclcpp::Node& node) {
                map_resolution = node.declare_parameter<double>(
                    "small_point_lio.map_resolution",
                    map_resolution
                );
                extrinsic_est_en = node.declare_parameter<bool>(
                    "small_point_lio.extrinsic_est_en",
                    extrinsic_est_en
                );
                laser_point_cov = node.declare_parameter<double>(
                    "small_point_lio.laser_point_cov",
                    laser_point_cov
                );
                laser_distance_cov_ratio = node.declare_parameter<double>(
                    "small_point_lio.laser_distance_cov_ratio",
                    laser_distance_cov_ratio
                );
                imu_meas_acc_cov = node.declare_parameter<double>(
                    "small_point_lio.imu_meas_acc_cov",
                    imu_meas_acc_cov
                );
                imu_meas_omg_cov = node.declare_parameter<double>(
                    "small_point_lio.imu_meas_omg_cov",
                    imu_meas_omg_cov
                );
                velocity_cov =
                    node.declare_parameter<double>("small_point_lio.velocity_cov", velocity_cov);
                acceleration_cov = node.declare_parameter<double>(
                    "small_point_lio.acceleration_cov",
                    acceleration_cov
                );
                omg_cov = node.declare_parameter<double>("small_point_lio.omg_cov", omg_cov);
                ba_cov = node.declare_parameter<double>("small_point_lio.ba_cov", ba_cov);
                bg_cov = node.declare_parameter<double>("small_point_lio.bg_cov", bg_cov);
                plane_threshold = node.declare_parameter<double>(
                    "small_point_lio.plane_threshold",
                    plane_threshold
                );
                match_sqaured =
                    node.declare_parameter<double>("small_point_lio.match_sqaured", match_sqaured);
                curv_threshold = node.declare_parameter<double>(
                    "small_point_lio.curv_threshold",
                    curv_threshold
                );
                check_satu = node.declare_parameter<bool>("small_point_lio.check_satu", check_satu);
                satu_acc = node.declare_parameter<double>("small_point_lio.satu_acc", satu_acc);
                satu_gyro = node.declare_parameter<double>("small_point_lio.satu_gyro", satu_gyro);
                acc_norm = node.declare_parameter<double>("small_point_lio.acc_norm", acc_norm);
                std::vector<double> extrinsic_T_vec = node.declare_parameter<std::vector<double>>(
                    "small_point_lio.extrinsic_T",
                    { 0.0, 0.0, 0.0 }
                );
                extrinsic_T =
                    Eigen::Vector3d(extrinsic_T_vec[0], extrinsic_T_vec[1], extrinsic_T_vec[2]);
                std::vector<double> extrinsic_R_vec = node.declare_parameter<std::vector<double>>(
                    "small_point_lio.extrinsic_R",
                    { 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0 }
                );
                extrinsic_R << extrinsic_R_vec[0], extrinsic_R_vec[1], extrinsic_R_vec[2],
                    extrinsic_R_vec[3], extrinsic_R_vec[4], extrinsic_R_vec[5], extrinsic_R_vec[6],
                    extrinsic_R_vec[7], extrinsic_R_vec[8];
                std::vector<double> gravity_vec = node.declare_parameter<std::vector<double>>(
                    "small_point_lio.gravity",
                    { 0.0, 0.0, -9.8 }
                );
                gravity = Eigen::Vector3d(gravity_vec[0], gravity_vec[1], gravity_vec[2]);
                init_map_size =
                    node.declare_parameter<int>("small_point_lio.init_map_size", init_map_size);
                fix_gravity_direction = node.declare_parameter<bool>(
                    "small_point_lio.fix_gravity_direction",
                    fix_gravity_direction
                );
                max_iter = node.declare_parameter<int>("small_point_lio.max_iter", max_iter);
            }
        } estimator_params;
        void load(rclcpp::Node& node) {
            preprocess_params.load(node);
            estimator_params.load(node);
        }
    } small_point_lio_params;
};
} // namespace rose_lm