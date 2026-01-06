#include "rose_lm.hpp"
#include "io/pcd_io.h"
#include "lidar_adapter/base_lidar.h"
#include "lidar_adapter/custom_mid360_driver.h"
#include "lidar_adapter/livox_custom_msg.h"
#include "lidar_adapter/livox_pointcloud2.h"
#include "lidar_adapter/unitree_lidar.h"
#include "lidar_adapter/velodyne_pointcloud2.h"
#include "mapping/pcd_mapping.h"
#include "parameters.hpp"
#include "rose_lm/common.hpp"
#include "small_point_lio/small_point_lio.hpp"
#include <nav_msgs/msg/odometry.hpp>
#include <param_deliver.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.hpp>
#include <tf2_ros/transform_listener.h>
namespace rose_lm {
struct RoseLm::Impl {
public:
    Impl(rclcpp::Node& node): tf_buffer_(node.get_clock()), tf_broadcaster_(node) {
        node_ = &node;
        params_.load(node);
        small_point_lio_ = small_point_lio::SmallPointLio::create(params_);
        small_point_lio_->setOdometryCallBack([this](const common::Odometry& odom) {
            publishOdometry(odom);
        });
        small_point_lio_->setPointCloudCallBack([this](const std::vector<common::Point>& pointcloud
                                                ) { publishPointCloud(pointcloud); });
        lidar_frame_ = node_->declare_parameter<std::string>("lidar_frame", "livox");
        robot_base_frame_ = node_->declare_parameter<std::string>("robot_base_frame", "base_link");
        base_frame_ = node_->declare_parameter<std::string>("base_frame", "base_link");
        save_pcd_ = node_->declare_parameter<bool>("save_pcd");
        if (save_pcd_) {
            RCLCPP_INFO(
                rclcpp::get_logger("rose_lm"),
                "enable save pcd "
            );
            pcd_mapping = std::make_unique<mapping::PCDMapping>(0.02);
        }
        std::string imu_topic = node_->declare_parameter("imu_topic", "livox/imu");
        imu_sub_ = node_->create_subscription<sensor_msgs::msg::Imu>(
            imu_topic,
            rclcpp::SensorDataQoS(),
            [this](const sensor_msgs::msg::Imu::SharedPtr msg) {
                common::ImuMsg imu_msg;
                imu_msg.angular_velocity = Eigen::Vector3d(
                    msg->angular_velocity.x,
                    msg->angular_velocity.y,
                    msg->angular_velocity.z
                );
                imu_msg.linear_acceleration = Eigen::Vector3d(
                    msg->linear_acceleration.x,
                    msg->linear_acceleration.y,
                    msg->linear_acceleration.z
                );
                imu_msg.timestamp = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
                small_point_lio_->preprocess_->on_imu_callback(imu_msg);
                small_point_lio_->handleOnce();
            }
        );
        std::string lidar_topic = node_->declare_parameter("lidar_topic", "livox/lidar");
        std::string lidar_type = node_->declare_parameter<std::string>("lidar_type");
        if (lidar_type == "livox_custom_msg") {
#ifdef HAVE_LIVOX_DRIVER
            lidar_adapter = std::make_unique<LivoxCustomMsgAdapter>();
#else
            RCLCPP_ERROR(
                rclcpp::get_logger("rose_lm"),
                "livox_custom_msg requested but not available!"
            );
            rclcpp::shutdown();
            return;
#endif
        } else if (lidar_type == "livox_pointcloud2") {
            lidar_adapter = std::make_unique<LivoxPointCloud2Adapter>();
        } else if (lidar_type == "custom_mid360_driver") {
            lidar_adapter = std::make_unique<CustomMid360DriverAdapter>();
        } else if (lidar_type == "unilidar") {
            lidar_adapter = std::make_unique<UnilidarAdapter>();
        } else if (lidar_type == "velodyne") {
            lidar_adapter = std::make_unique<VelodynePointCloud2>();
        } else {
            RCLCPP_ERROR(rclcpp::get_logger("rose_lm"), "unknwon lidar type");
            rclcpp::shutdown();
            return;
        }
        lidar_adapter->setup_subscription(
            node_,
            lidar_topic,
            [this](const std::vector<common::Point>& pointcloud) {
                static int stat_count = 0;
                static double stat_cost = 0.0;
                static auto window_start = std::chrono::system_clock::now();

                auto start = std::chrono::system_clock::now();

                small_point_lio_->preprocess_->on_point_cloud_callback(pointcloud);
                small_point_lio_->handleOnce();

                auto end = std::chrono::system_clock::now();
                double cost_ms = std::chrono::duration<double, std::milli>(end - start).count();

                stat_count++;
                stat_cost += cost_ms;

                double elapsed_ms =
                    std::chrono::duration<double, std::milli>(end - window_start).count();

                if (elapsed_ms >= 1000.0) {
                    long ts = std::chrono::duration_cast<std::chrono::seconds>(
                                  window_start.time_since_epoch()
                    )
                                  .count();

                    RCLCPP_INFO_STREAM(
                        rclcpp::get_logger("rose_lm"),
                        "[Lidar Stat] Time: " << ts << " | FPS: " << stat_count
                                              << " | Total Cost: " << stat_cost << " ms"
                                              << " | Processed Points: "
                                              << small_point_lio_->processed_points
                    );

                    window_start = end;
                    stat_count = 0;
                    stat_cost = 0.0;
                    small_point_lio_->processed_points = 0;
                }
            }
        );
        odom_pub_ = node_->create_publisher<nav_msgs::msg::Odometry>("/Odometry", 1000);
        pointcloud_pub_ =
            node_->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_registered", 1000);
        map_save_trigger_ = node_->create_service<std_srvs::srv::Trigger>(
            "map_save",
            [this](
                const std_srvs::srv::Trigger::Request::SharedPtr req,
                std_srvs::srv::Trigger::Response::SharedPtr res
            ) {
                saveMap(req, res);
            }
        );
    }
    static Ptr create(rclcpp::Node& node) {
        return std::make_unique<RoseLm>(node);
    }
    void saveMap(
        const std_srvs::srv::Trigger::Request::SharedPtr req,
        std_srvs::srv::Trigger::Response::SharedPtr res
    ) {
        if (!save_pcd_) {
            RCLCPP_ERROR(rclcpp::get_logger("rose_lm"), "pcd save is disabled");
            return;
        }
        RCLCPP_INFO(rclcpp::get_logger("rose_lm"), "waiting for pcd saving ...");
        auto pointcloud_to_save = std::make_shared<std::vector<Eigen::Vector3f>>();
        *pointcloud_to_save = pcd_mapping->get_points();
        std::thread([pointcloud_to_save]() {
            io::pcd::write_pcd(ROOT_DIR + "/pcd/scan.pcd", *pointcloud_to_save);
            RCLCPP_INFO(rclcpp::get_logger("rose_lm"), "save pcd success");
        }).detach();
    }

    void publishPointCloud(const std::vector<common::Point>& pointcloud) {
        if (pointcloud_pub_->get_subscription_count() > 0) {
            builtin_interfaces::msg::Time time_msg;
            time_msg.sec = std::floor(last_odometry_.timestamp);
            time_msg.nanosec =
                static_cast<uint32_t>((last_odometry_.timestamp - time_msg.sec) * 1e9);
            geometry_msgs::msg::TransformStamped lidar_frame_to_base_link_transform;
            tf2::Transform tf_in_odom = tf_odom_to_lodom_;
            lidar_frame_to_base_link_transform.transform = tf2::toMsg(tf_in_odom);
            lidar_frame_to_base_link_transform.header.frame_id = "odom";
            Eigen::Vector3f lidar_frame_to_base_link_T;
            lidar_frame_to_base_link_T
                << static_cast<float>(lidar_frame_to_base_link_transform.transform.translation.x),
                static_cast<float>(lidar_frame_to_base_link_transform.transform.translation.y),
                static_cast<float>(lidar_frame_to_base_link_transform.transform.translation.z);
            Eigen::Matrix3f lidar_frame_to_base_link_R =
                Eigen::Quaternionf(
                    static_cast<float>(lidar_frame_to_base_link_transform.transform.rotation.w),
                    static_cast<float>(lidar_frame_to_base_link_transform.transform.rotation.x),
                    static_cast<float>(lidar_frame_to_base_link_transform.transform.rotation.y),
                    static_cast<float>(lidar_frame_to_base_link_transform.transform.rotation.z)
                )
                    .toRotationMatrix();
            sensor_msgs::msg::PointCloud2 msg;
            msg.header.stamp = time_msg;
            msg.header.frame_id = "odom";
            msg.width = pointcloud.size();
            msg.height = 1;
            msg.fields.reserve(4);
            sensor_msgs::msg::PointField field;
            field.name = "x";
            field.offset = 0;
            field.datatype = sensor_msgs::msg::PointField::FLOAT32;
            field.count = 1;
            msg.fields.push_back(field);
            field.name = "y";
            field.offset = 4;
            field.datatype = sensor_msgs::msg::PointField::FLOAT32;
            field.count = 1;
            msg.fields.push_back(field);
            field.name = "z";
            field.offset = 8;
            field.datatype = sensor_msgs::msg::PointField::FLOAT32;
            field.count = 1;
            msg.fields.push_back(field);
            field.name = "intensity";
            field.offset = 12;
            field.datatype = sensor_msgs::msg::PointField::FLOAT32;
            field.count = 1;
            msg.fields.push_back(field);
            msg.is_bigendian = false;
            msg.point_step = 16;
            msg.row_step = msg.width * msg.point_step;
            msg.data.resize(msg.row_step * msg.height);
            Eigen::Vector3f transformed_point;
            auto pointer = reinterpret_cast<float*>(msg.data.data());
            for (const auto& point: pointcloud) {
                transformed_point =
                    lidar_frame_to_base_link_R * point.position + lidar_frame_to_base_link_T;
                *pointer = transformed_point.x();
                ++pointer;
                *pointer = transformed_point.y();
                ++pointer;
                *pointer = transformed_point.z();
                ++pointer;
                *pointer = (point.position - last_odometry_.position.cast<float>()).norm();
                ++pointer;
                if (save_pcd_)
                    pcd_mapping->add_point(transformed_point);
            }
            msg.is_dense = false;
            pointcloud_pub_->publish(msg);
        } else if (save_pcd_) {
            geometry_msgs::msg::TransformStamped lidar_frame_to_base_link_transform;
            tf2::Transform tf_in_odom = tf_odom_to_lodom_;
            lidar_frame_to_base_link_transform.transform = tf2::toMsg(tf_in_odom);
            Eigen::Vector3f lidar_frame_to_base_link_T;
            lidar_frame_to_base_link_T
                << static_cast<float>(lidar_frame_to_base_link_transform.transform.translation.x),
                static_cast<float>(lidar_frame_to_base_link_transform.transform.translation.y),
                static_cast<float>(lidar_frame_to_base_link_transform.transform.translation.z);
            Eigen::Matrix3f lidar_frame_to_base_link_R =
                Eigen::Quaternionf(
                    static_cast<float>(lidar_frame_to_base_link_transform.transform.rotation.w),
                    static_cast<float>(lidar_frame_to_base_link_transform.transform.rotation.x),
                    static_cast<float>(lidar_frame_to_base_link_transform.transform.rotation.y),
                    static_cast<float>(lidar_frame_to_base_link_transform.transform.rotation.z)
                )
                    .toRotationMatrix();
            Eigen::Vector3f transformed_point;
            for (const auto& point: pointcloud) {
                transformed_point =
                    lidar_frame_to_base_link_R * point.position + lidar_frame_to_base_link_T;
                pcd_mapping->add_point(transformed_point);
            }
        }
    }
    void publishOdometry(const common::Odometry& odometry) {
        auto lookupTF = [this](
                            const std::string& source,
                            const std::string& target,
                            const rclcpp::Time& stamp
                        ) -> tf2::Transform {
            try {
                auto t = tf_buffer_.lookupTransform(
                    source,
                    target,
                    stamp,
                    rclcpp::Duration::from_seconds(0.05)
                );

                tf2::Transform tf;
                tf2::fromMsg(t.transform, tf);
                return tf;
            } catch (tf2::TransformException& ex) {
                RCLCPP_WARN(
                    rclcpp::get_logger("rose_lm"),
                    "TF lookup failed (%s->%s): %s",
                    source.c_str(),
                    target.c_str(),
                    ex.what()
                );
                return tf2::Transform::getIdentity();
            }
        };

        last_odometry_ = odometry;
        builtin_interfaces::msg::Time time_msg;
        time_msg.sec = std::floor(odometry.timestamp);
        time_msg.nanosec = static_cast<uint32_t>((odometry.timestamp - time_msg.sec) * 1e9);
        tf2::Transform tf_lodom_to_lidar;
        tf_lodom_to_lidar.setOrigin(
            tf2::Vector3(odometry.position.x(), odometry.position.y(), odometry.position.z())
        );
        tf_lodom_to_lidar.setRotation(tf2::Quaternion(
            odometry.orientation.x(),
            odometry.orientation.y(),
            odometry.orientation.z(),
            odometry.orientation.w()
        ));

        if (is_first_pointcloud_) {
            tf2::Transform l2r = lookupTF(lidar_frame_, robot_base_frame_, time_msg);
            tf_odom_to_lodom_ = l2r.inverse();
            is_first_pointcloud_ = false;
        }

        tf2::Transform tf_lidar_to_robot = lookupTF(lidar_frame_, robot_base_frame_, time_msg);

        tf2::Transform tf_lidar_to_base = lookupTF(lidar_frame_, base_frame_, time_msg);

        tf2::Transform tf_odom_to_base = tf_odom_to_lodom_ * tf_lodom_to_lidar * tf_lidar_to_base;
        tf2::Transform tf_odom_to_robot = tf_odom_to_lodom_ * tf_lodom_to_lidar * tf_lidar_to_robot;
        geometry_msgs::msg::TransformStamped tf_msg;
        tf_msg.header.stamp = time_msg;
        tf_msg.header.frame_id = "odom";
        tf_msg.child_frame_id = base_frame_;
        tf_msg.transform = tf2::toMsg(tf_odom_to_base);
        tf_broadcaster_.sendTransform(tf_msg);

        nav_msgs::msg::Odometry odom_msg;
        odom_msg.header.stamp = time_msg;
        odom_msg.header.frame_id = "odom";
        odom_msg.child_frame_id = robot_base_frame_;

        const auto& t = tf_odom_to_robot.getOrigin();
        const auto& q = tf_odom_to_robot.getRotation();
        odom_msg.pose.pose.position.x = t.x();
        odom_msg.pose.pose.position.y = t.y();
        odom_msg.pose.pose.position.z = t.z();
        odom_msg.pose.pose.orientation = tf2::toMsg(q);

        static tf2::Transform last_tf;
        static rclcpp::Time last_stamp = rclcpp::Time(time_msg);
        if (last_stamp.nanoseconds() > 0) {
            rclcpp::Time current_time(time_msg);
            double dt = (current_time - last_stamp).seconds();
            if (dt > 1e-6) {
                auto diff = tf_odom_to_robot.getOrigin() - last_tf.getOrigin();
                odom_msg.twist.twist.linear.x = diff.x() / dt;
                odom_msg.twist.twist.linear.y = diff.y() / dt;
                odom_msg.twist.twist.linear.z = diff.z() / dt;

                tf2::Quaternion dq =
                    tf_odom_to_robot.getRotation() * last_tf.getRotation().inverse();
                tf2::Vector3 axis = dq.getAxis();
                double angle = dq.getAngle();
                tf2::Vector3 ang_vel = axis * angle / dt;
                odom_msg.twist.twist.angular.x = ang_vel.x();
                odom_msg.twist.twist.angular.y = ang_vel.y();
                odom_msg.twist.twist.angular.z = ang_vel.z();
            }
        }
        last_tf = tf_odom_to_robot;
        last_stamp = time_msg;
        odom_pub_->publish(odom_msg);
    }

    bool save_pcd_;
    common::Odometry last_odometry_;
    tf2::Transform tf_odom_to_lodom_;
    bool is_first_pointcloud_ = true;
    Parameters params_;
    small_point_lio::SmallPointLio::Ptr small_point_lio_;
    std::string lidar_frame_;
    std::string robot_base_frame_;
    std::string base_frame_;

    std::unique_ptr<mapping::PCDMapping> pcd_mapping;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    std::unique_ptr<LidarAdapterBase> lidar_adapter;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_pub_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr map_save_trigger_;
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_ { tf_buffer_ };
    tf2_ros::TransformBroadcaster tf_broadcaster_;
    rclcpp::Node* node_;
};
RoseLm::RoseLm(rclcpp::Node& node) {
    impl_ = std::make_unique<Impl>(node);
}
RoseLm::~RoseLm() {
    impl_.reset();
}
} // namespace rose_lm