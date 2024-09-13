import os
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    pkg_share = get_package_share_directory("multi_lidar_calibrator")
    parameter_file = os.path.join(pkg_share, "config", "params.yaml")
    output = os.path.join(pkg_share, "output")

    # Declare launch arguments
    output_dir_arg = DeclareLaunchArgument(
        "output_dir", default_value=output, description="Path to the output directory."
    )

    params_declare = DeclareLaunchArgument(
        "parameter_file",
        default_value=parameter_file,
        description="Path to the ROS2 parameters file to use.",
    )

    # Define lidar topics for each node instance
    left_to_center_lidar_calibration_node_topics = ["/center_lidar/lidar_points", "/left_lidar/lidar_points"]
    right_to_center_lidar_calibration_node_topics = ["/center_lidar/lidar_points", "/right_lidar/lidar_points"]

    # LaunchConfiguration to fetch the values of the arguments
    parameter_file_launch_config = LaunchConfiguration("parameter_file")
    output_dir_launch_config = LaunchConfiguration("output_dir")

    # Node 1 (center and left lidar)
    left_to_center_calibration_node = Node(
        package="multi_lidar_calibrator",
        executable="multi_lidar_calibrator",
        name="multi_lidar_calibration_node_1",
        parameters=[
            parameter_file_launch_config,
            {
                'output_dir': output_dir_launch_config,
                'lidar_topics': left_to_center_lidar_calibration_node_topics
            }
        ],
        remappings=[('/lidar_calibration', '/offline_calibration/start_left_lidar_calibration')],
        output="screen",
    )

    # Node 2 (center and right lidar)
    right_to_center_calibration_node = Node(
        package="multi_lidar_calibrator",
        executable="multi_lidar_calibrator",
        name="multi_lidar_calibration_node_2",
        parameters=[
            parameter_file_launch_config,
            {
                'output_dir': output_dir_launch_config,
                'lidar_topics': right_to_center_lidar_calibration_node_topics
            }
        ],
        remappings=[('/lidar_calibration', '/offline_calibration/start_right_lidar_calibration')],
        output="screen",
    )

    return LaunchDescription(
        [
            params_declare,
            output_dir_arg,
            left_to_center_calibration_node,
            right_to_center_calibration_node,
        ]
    )
