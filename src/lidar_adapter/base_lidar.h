/**
 * This file is part of Small Point-LIO, an advanced Point-LIO algorithm implementation.
 * Copyright (C) 2025  Yingjie Huang
 * Licensed under the MIT License. See License.txt in the project root for license information.
 */

#pragma once

#include "rose_lm/common.hpp"
#include <rclcpp/node.hpp>

namespace rose_lm {

class LidarAdapterBase {
public:
    virtual ~LidarAdapterBase() = default;
    virtual void setup_subscription(
        rclcpp::Node* node,
        const std::string& topic,
        std::function<void(const std::vector<common::Point>&)> callback
    ) = 0;
};

} // namespace rose_lm
