#pragma once
#include "estimator.h"
#include "preprocess.hpp"
namespace rose_lm {
namespace small_point_lio {

    class SmallPointLio {
    public:
        using OdometryCallBack = std::function<void(const common::Odometry&)>;
        using PointCloudCallBack = std::function<void(const std::vector<common::Point>&)>;
        using Ptr = std::shared_ptr<SmallPointLio>;
        SmallPointLio(Parameters params): params_(params) {
            preprocess_ = Preprocess::create(params_);
            estimator_ = Estimator::create(params_);
            Q = estimator_->process_noise_cov();
            estimator_->reset();
            preprocess_->reset();
        }
        static Ptr create(Parameters params) {
            return std::make_shared<SmallPointLio>(params);
        }
        void handleOnce() {
            if (!is_init_) {
                int total_points = 0;
                for (const auto& b: preprocess_->point_batch_deque) {
                    total_points += b.points.size();
                }
                if ((!preprocess_->imu_deque.empty() || !preprocess_->point_batch_deque.empty())
                    && total_points >= params_.small_point_lio_params.estimator_params.init_map_size
                    && (!params_.small_point_lio_params.estimator_params.fix_gravity_direction
                        || preprocess_->imu_deque.size() >= 200))
                {
                    for (const auto& batch: preprocess_->point_batch_deque) {
                        for (const auto& point: batch.points) {
                            estimator_->ivox->add_point(point.position);
                        }
                    }
                    // fix gravity direction
                    if (params_.small_point_lio_params.estimator_params.fix_gravity_direction) {
                        estimator_->kf.x.gravity = Eigen::Matrix<state::value_type, 3, 1>::Zero();
                        for (const auto& imu_msg: preprocess_->imu_deque) {
                            estimator_->kf.x.gravity +=
                                imu_msg.linear_acceleration.cast<state::value_type>();
                        }
                        state::value_type scale =
                            -static_cast<state::value_type>(
                                params_.small_point_lio_params.estimator_params.gravity.norm()
                            )
                            / estimator_->kf.x.gravity.norm();
                        estimator_->kf.x.gravity *= scale;

                    } else {
                        estimator_->kf.x.gravity = params_.small_point_lio_params.estimator_params
                                                       .gravity.cast<state::value_type>();
                    }
                    std::cout << "gravity: " << estimator_->kf.x.gravity.transpose() << std::endl;
                    estimator_->kf.x.acceleration = -estimator_->kf.x.gravity;
                    // init time
                    if (preprocess_->point_batch_deque.empty()) {
                        time_current_ = preprocess_->imu_deque.back().timestamp;
                    } else if (preprocess_->imu_deque.empty()) {
                        time_current_ = preprocess_->point_batch_deque.back().timestamp;
                    } else {
                        time_current_ = std::max(
                            preprocess_->point_batch_deque.back().timestamp,
                            preprocess_->imu_deque.back().timestamp
                        );
                    }
                    estimator_->kf.init_timestamp(time_current_);
                    // clear data
                    preprocess_->dense_point_deque.clear();
                    preprocess_->point_batch_deque.clear();
                    preprocess_->imu_deque.clear();
                    is_init_ = true;
                }
                return;
            }

            bool is_publish_odometry = !preprocess_->imu_deque.empty()
                && !preprocess_->point_batch_deque.empty()
                && preprocess_->point_batch_deque.front().timestamp
                    < preprocess_->imu_deque.back().timestamp
                && preprocess_->point_batch_deque.back().timestamp
                    > preprocess_->imu_deque.front().timestamp;
            while (!preprocess_->imu_deque.empty() && !preprocess_->point_batch_deque.empty()) {
                const common::Batch& batch_frame = preprocess_->point_batch_deque.front();
                const common::ImuMsg& imu_msg = preprocess_->imu_deque.front();
                if (batch_frame.timestamp < imu_msg.timestamp) {
                    // point update
                    if (batch_frame.timestamp < time_current_) {
                        preprocess_->point_batch_deque.pop_front();
                        continue;
                    }
                    double t_ref = batch_frame.timestamp;
                    time_current_ = t_ref;
                    // predict
                    estimator_->kf.predict_state(time_current_);
                    estimator_->current_batch = batch_frame;
                    // update
                    estimator_->kf.update_iterated_batch();
                    for (const auto& point: estimator_->points_odom_frame) {
                        estimator_->ivox->add_point(point);
                        common::Point point_to_add;
                        point_to_add.position = point;
                        pointcloud_odom_frame_.emplace_back(point_to_add);
                    }
                    preprocess_->point_batch_deque.pop_front();
                    used_points += estimator_->processed_point_num;
                    processed_points += batch_frame.points.size();
                } else {
                    // imu update
                    if (imu_msg.timestamp < time_current_) {
                        preprocess_->imu_deque.pop_front();
                        continue;
                    }
                    time_current_ = imu_msg.timestamp;

                    // predict
                    estimator_->kf.predict_state(time_current_);
                    estimator_->kf.predict_cov(time_current_, Q);

                    // update
                    estimator_->angular_velocity =
                        imu_msg.angular_velocity.cast<state::value_type>();
                    estimator_->linear_acceleration =
                        imu_msg.linear_acceleration.cast<state::value_type>();
                    estimator_->kf.update_imu();

                    preprocess_->imu_deque.pop_front();
                }
            }
            auto hat = [](const Eigen::Vector3d& v) {
                return (Eigen::Matrix3d() << 0, -v.z(), v.y(), v.z(), 0, -v.x(), -v.y(), v.x(), 0)
                    .finished();
            };
            for (auto& pt: preprocess_->dense_point_deque) {
                const auto& x = estimator_->kf.x;
                const Eigen::Matrix3d R_LI =
                    params_.small_point_lio_params.estimator_params.extrinsic_est_en
                    ? x.offset_R_L_I
                    : params_.small_point_lio_params.estimator_params.extrinsic_R;
                const Eigen::Vector3d T_LI =
                    params_.small_point_lio_params.estimator_params.extrinsic_est_en
                    ? x.offset_T_L_I
                    : params_.small_point_lio_params.estimator_params.extrinsic_T;
                const Eigen::Vector3d w = x.omg;
                const double wn = w.norm();

                double dt = pt.timestamp - time_current_;
                Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
                if (wn > 1e-8)
                    R += hat(w / wn * (wn * dt));

                Eigen::Vector3d p = R_LI * pt.position.cast<double>() + T_LI;
                common::Point point;
                point.position =
                    ((x.rotation * R) * p + x.position + x.velocity * dt).cast<float>();
                //pointcloud_odom_frame_.emplace_back(point);
            }
            preprocess_->dense_point_deque.clear();
            if (is_publish_odometry) {
                publishOdometry(time_current_);

                if (!pointcloud_odom_frame_.empty()) {
                    publishPointCloud(pointcloud_odom_frame_);

                    pointcloud_odom_frame_.clear();
                }
            }
        }
        void setOdometryCallBack(OdometryCallBack cb) {
            odom_cb = cb;
        }
        void setPointCloudCallBack(PointCloudCallBack cb) {
            pc_cb = cb;
        }
        void publishPointCloud(const std::vector<common::Point>& pointcloud) {
            if (pc_cb) {
                pc_cb(pointcloud);
            }
        }
        void publishOdometry(double timestamp) {
            common::Odometry odometry;
            odometry.timestamp = timestamp;
            odometry.position = estimator_->kf.x.position.cast<double>();
            odometry.velocity = estimator_->kf.x.velocity.cast<double>();
            odometry.orientation = estimator_->kf.x.rotation.cast<double>();
            odometry.angular_velocity = estimator_->kf.x.omg.cast<double>();
            if (odom_cb) {
                odom_cb(odometry);
            }
        }

        OdometryCallBack odom_cb;
        PointCloudCallBack pc_cb;
        Eigen::Matrix<state::value_type, state::DIM, state::DIM> Q;
        double time_current_ = 0.0;
        Preprocess::Ptr preprocess_;
        Estimator::Ptr estimator_;
        Parameters params_;
        bool is_init_ = false;
        std::vector<common::Point> pointcloud_odom_frame_;
        int processed_points = 0;
        int used_points = 0;
    };

} // namespace small_point_lio

} // namespace rose_lm