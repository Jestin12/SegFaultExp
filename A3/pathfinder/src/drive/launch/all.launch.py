from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get the paths to the packages
    drive_dir = get_package_share_directory('drive')
    pedestrian_dir = get_package_share_directory('pedestrian')

    # Include the first launch file from package_a
    driver_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(drive_dir, 'launch', 'drive.launch.py')
        )
    )

    # Include the second launch file from package_b
    pedestrian_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pedestrian_dir, 'launch', 'pedestrian.launch.py')
        )
    )

    delayed_pedestrian_launch = TimerAction(
        period=15.0,  # Delay in seconds (e.g., 5 seconds)
        actions=[pedestrian_launch]
    )

    return LaunchDescription([
        driver_launch,
        delayed_pedestrian_launch,
    ])