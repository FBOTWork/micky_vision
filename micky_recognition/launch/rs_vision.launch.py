import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node

def generate_launch_description():
    declared_arguments = [
        DeclareLaunchArgument(
            "use_rviz",
            default_value="false",
            description="Start RViz as a parameter with this initialization file.",
        )
    ]
    use_rviz = LaunchConfiguration("use_rviz")

    # 1. Launch da câmera RealSense
    launch_camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('realsense2_camera'),
                'launch', 'rs_launch.py'
            )
        )
    )

    # 2. RViz
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", os.path.join(get_package_share_directory('micky_recognition'), 'config/rviz/reference.rviz')],
        condition=IfCondition(use_rviz),
    )

    return LaunchDescription(
        declared_arguments + [
            launch_camera,
            rviz_node
        ]
    )