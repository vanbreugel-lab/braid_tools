from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'braid_url',
            default_value='http://134.197.37.229:8397/',
            description='URL of the Braid model server'),
        Node(
            package='braid_tools',
            executable='braid_ros_listener.py',
            name='braid_ros_listener',
            output='screen',
            arguments=['--braid-model-server-url', LaunchConfiguration('braid_url')],
        ),
    ])
