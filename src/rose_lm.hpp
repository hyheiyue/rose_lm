#pragma once

#include <memory>
#include <rclcpp/node.hpp>
namespace rose_lm {
class RoseLm {
public:
    using Ptr = std::unique_ptr<RoseLm>;
    explicit RoseLm(rclcpp::Node& node);
    ~RoseLm();
    static Ptr create(rclcpp::Node& node) {
        return std::make_unique<RoseLm>(node);
    }

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace rose_lm