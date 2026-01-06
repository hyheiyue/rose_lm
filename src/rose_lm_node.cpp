#include "rose_lm.hpp"
#include <rclcpp/rclcpp.hpp>
namespace rose_lm {
class RoseLmNode: public rclcpp::Node {
public:
    RoseLmNode(const rclcpp::NodeOptions& options): Node("rose_lm_node", options) {
        rose_lm_ = RoseLm::create(*this);
    }
    RoseLm::Ptr rose_lm_;
};
} // namespace rose_lm
#include "rclcpp_components/register_node_macro.hpp"

// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable when its library
// is being loaded into a running process.
RCLCPP_COMPONENTS_REGISTER_NODE(rose_lm::RoseLmNode)
