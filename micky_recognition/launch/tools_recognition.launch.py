#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from launch import LaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from launch_ros.substitutions import FindPackageShare
import os

def generate_launch_description():

    config_file_path = PathJoinSubstitution([
        get_package_share_directory('micky_recognition'),
        'config',
        'yolov8_tools_recognition.yaml']
    )

    declared_arguments = []
    declared_arguments.append(
        DeclareLaunchArgument(
            'config_object_recognition',
            default_value=config_file_path,
            description='Path to the parameter file'
        ))
    
    declared_arguments.append(
        DeclareLaunchArgument(
            'use_realsense',
            default_value='true',
            description="If it should run the realsense node"
        ))

    tools_recognition_node = Node(
        package='micky_recognition',
        executable='tools_recognition',
        name='tools_recognition',
        parameters=[LaunchConfiguration('config_object_recognition'),],
    )

    realsense2_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('realsense2_camera'), 'launch', 'rs_launch.py')
        ),
        launch_arguments={
            'camera_name': 'camera',
            'camera_namespace': 'micky_vision',
            'enable_rgbd': 'true',
            'enable_sync': 'true',
            'align_depth.enable': 'true',
            'enable_color': 'true',
            'enable_depth': 'true',
            'pointcloud.enable': 'true'
        }.items(),
        condition=IfCondition(LaunchConfiguration('use_realsense'))
    )

    return LaunchDescription([
        *declared_arguments,
        tools_recognition_node,
        realsense2_node
    ])