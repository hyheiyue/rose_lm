/**
 * This file is part of Small Point-LIO, an advanced Point-LIO algorithm implementation.
 * Copyright (C) 2025  Yingjie Huang
 * Licensed under the MIT License. See License.txt in the project root for license information.
 */

#pragma once

#include "base_lidar.h"
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

namespace rose_lm {

class VelodynePointCloud2: public LidarAdapterBase {
private:
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription;

public:
    inline void setup_subscription(
        rclcpp::Node* node,
        const std::string& topic,
        std::function<void(const std::vector<common::Point>&)> callback
    ) override {
        subscription = node->create_subscription<sensor_msgs::msg::PointCloud2>(
            topic,
            rclcpp::SensorDataQoS(),
            [callback](const sensor_msgs::msg::PointCloud2& msg) {
                double base_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9;

                sensor_msgs::PointCloud2ConstIterator<float> iter_x(msg, "x");
                sensor_msgs::PointCloud2ConstIterator<float> iter_y(msg, "y");
                sensor_msgs::PointCloud2ConstIterator<float> iter_z(msg, "z");

                sensor_msgs::PointCloud2ConstIterator<float> iter_timestamp(msg, "time");

                size_t size = msg.width * msg.height;
                std::vector<common::Point> pointcloud;
                pointcloud.reserve(size);

                for (size_t i = 0; i < size; ++i) {
                    common::Point p;

                    p.position << *iter_x, *iter_y, *iter_z;

                    float relative_t = *iter_timestamp * 0.0001;
                    p.timestamp = base_time + relative_t;

                    pointcloud.push_back(p);

                    ++iter_x;
                    ++iter_y;
                    ++iter_z;
                    ++iter_timestamp;
                }

                callback(pointcloud);
            }
        );
    }
};

} // namespace rose_lm
