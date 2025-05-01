from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='drive',
            namespace='driver',
            executable='driver',
            name='driver'
        ),
        Node(
            package='drive',
            namespace='coordinates',
            executable='coordinates',
            name='CoordinatesFinder'
        )
    ])