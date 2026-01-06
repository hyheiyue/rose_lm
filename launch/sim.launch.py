from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    rose_lm_node = Node(
        package="rose_lm",
        executable="rose_lm_node",
        name="rose_lm_node",
        output="screen",
        parameters=[
            PathJoinSubstitution(
                [
                    FindPackageShare("rose_lm"),
                    "config",
                    "sim.yaml",
                ]
            )
        ],
    )

    return LaunchDescription([rose_lm_node])
