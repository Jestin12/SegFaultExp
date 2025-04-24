from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='pedestrian',
            namespace='pedestrian',
            executable='Pedestrian',
            name='pedestrian'
        )
    ])