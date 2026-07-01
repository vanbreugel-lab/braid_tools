from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'braid_url',
            default_value='http://134.197.37.229:8397/',
            description='URL of the Braid model server'),
        DeclareLaunchArgument(
            'plot_config',
            default_value=PathJoinSubstitution(
                [FindPackageShare('braid_tools'), 'plot_config', 'big_tunnel.yaml']),
            description='Path to a plot config yaml file'),
        Node(
            package='braid_tools',
            executable='braid_ros_listener.py',
            name='braid_ros_listener',
            output='screen',
            arguments=['--braid-model-server-url', LaunchConfiguration('braid_url')],
        ),
        Node(
            package='braid_tools',
            executable='braid_realtime_plotter.py',
            name='braid_realtime_plotter',
            output='screen',
            arguments=['--config', LaunchConfiguration('plot_config')],
        ),
    ])
