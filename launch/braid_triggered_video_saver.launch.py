from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_config = PathJoinSubstitution([FindPackageShare('braid_tools'),
                                           'video_config',
                                           'triggered_video_saver.toml'])

    return LaunchDescription([
        DeclareLaunchArgument(
            'config',
            default_value=default_config,
            description='Path to the triggered_video_saver .toml config',
        ),
        Node(
            package='braid_tools',
            executable='braid_triggered_video_saver.py',
            name='braid_triggered_video_saver',
            output='screen',
            arguments=['--config', LaunchConfiguration('config')],
        ),
    ])
