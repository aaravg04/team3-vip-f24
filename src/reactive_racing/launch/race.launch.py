# File format reference https://docs.ros.org/en/humble/Tutorials/Intermediate/Launch/Creating-Launch-Files.html

from launch import LaunchDescription
from launch_ros.actions import Node
# from launch.substitutions import Command
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from launch.actions import TimerAction
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    vesc_config = os.path.join(
        get_package_share_directory('f1tenth_stack'),
        'config',
        'vesc.yaml'
    )
    sensors_config = os.path.join(
        get_package_share_directory('f1tenth_stack'),
        'config',
        'sensors.yaml'
    )
    mux_config = os.path.join(
        get_package_share_directory('f1tenth_stack'),
        'config',
        'mux.yaml'
    )

    vesc_la = DeclareLaunchArgument(
        'vesc_config',
        default_value=vesc_config,
        description='Descriptions for vesc configs')
    sensors_la = DeclareLaunchArgument(
        'sensors_config',
        default_value=sensors_config,
        description='Descriptions for sensor configs')
    mux_la = DeclareLaunchArgument(
        'mux_config',
        default_value=mux_config,
        description='Descriptions for ackermann mux configs')

    ld = LaunchDescription([vesc_la, sensors_la, mux_la])

    ackermann_to_vesc_node = Node(
        package='vesc_ackermann',
        executable='ackermann_to_vesc_node',
        name='ackermann_to_vesc_node',
        output="screen",
        parameters=[LaunchConfiguration('vesc_config')]
    )
    vesc_driver_node = Node(
        package='vesc_driver',
        executable='vesc_driver_node',
        name='vesc_driver_node',
        output="screen",
        parameters=[LaunchConfiguration('vesc_config')]
    )
    ackermann_mux_node = Node(
        package='ackermann_mux',
        executable='ackermann_mux',
        name='ackermann_mux',
        output="screen",
        parameters=[LaunchConfiguration('mux_config')],
        remappings=[('ackermann_drive_out', 'ackermann_cmd')]
    )
    lidar_node = Node(
        package='urg_node',
        executable='urg_node_driver',
        name='urg_node',
        parameters=[LaunchConfiguration('sensors_config')]
    )

    perception_node = Node(
        package='reactive_racing',
        executable='perception',
        name='perception',
        output="screen",
    )
    ctrl_node = Node(
        package='reactive_racing',
        executable='control',
        name='control',
        output="screen",
    )

    zed2node = Node(
            package='reactive_racing',
            executable='zed2_custom_node',
            name='zed2_custom_node',
            output='screen',
            emulate_tty=True,
            parameters=[
                {'camera_resolution': 'HD720'},
                {'camera_fps': 30},
                {'depth_mode': 'ULTRA'}
            ]
        )
    
    # Add noded to LaunchDescription
    ld.add_action(ackermann_to_vesc_node)
    ld.add_action(vesc_driver_node)
    ld.add_action(ackermann_mux_node)
    ld.add_action(lidar_node)
    ld.add_action(perception_node)
    ld.add_action(zed2node)
    ld.add_action(TimerAction(
       period=1.0,
       actions=[ctrl_node]
    ))

    return ld