"""Full ROS 2 side of the triggered-video pipeline in one launch.

Starts: braid_ros_listener (3D tracking from the braid model server),
topic_relay_client (trigger messages from the ROS 1 machine),
braid_trigger_adapter (Float64MultiArray -> BraidTrigger), and
braid_triggered_video_saver (strand-cam recording). See
TRIGGERED_VIDEO_SYSTEM.md for the full both-machines guide.

    ros2 launch braid_tools braid_triggered_video_pipeline.launch.py \
        braid_url:=http://BRAID.MACHINE.IP:8397/ \
        relay_url:=http://BRAID.MACHINE.IP:8398/
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_video_config = PathJoinSubstitution([FindPackageShare('braid_tools'),
                                                 'video_config',
                                                 'triggered_video_saver.toml'])

    return LaunchDescription([
        DeclareLaunchArgument(
            'braid_url', default_value='http://134.197.37.229:8397/',
            description='Braid model server URL (tracking data)'),
        DeclareLaunchArgument(
            'relay_url', default_value='http://134.197.37.229:8398/',
            description='topic_relay_server URL on the ROS 1 machine'),
        DeclareLaunchArgument(
            'video_config', default_value=default_video_config,
            description='triggered_video_saver .toml'),
        DeclareLaunchArgument(
            'trigger_topic', default_value='braid_trigger_topic',
            description='relayed Float64MultiArray trigger topic (matches the '
                        'ROS 1 trigger node config yaml)'),

        Node(package='braid_tools', executable='braid_ros_listener.py',
             name='braid_ros_listener', output='screen',
             arguments=['--braid-model-server-url', LaunchConfiguration('braid_url')]),
        Node(package='braid_tools', executable='topic_relay_client.py',
             name='topic_relay_client', output='screen',
             arguments=['--url', LaunchConfiguration('relay_url')]),
        Node(package='braid_tools', executable='braid_trigger_adapter.py',
             name='braid_trigger_adapter', output='screen',
             arguments=['--trigger-topic', LaunchConfiguration('trigger_topic')]),
        Node(package='braid_tools', executable='braid_triggered_video_saver.py',
             name='braid_triggered_video_saver', output='screen',
             arguments=['--config', LaunchConfiguration('video_config')]),
    ])
