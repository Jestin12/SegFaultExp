from setuptools import setup

package_name = 'image_flipper'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'rclpy', 'sensor_msgs', 'cv_bridge', 'rosbag2_py'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email',
    description='A package to flip images in a ROS 2 bag file',
    license='Your License',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'flip_images = image_flipper.flip_images:main',
        ],
    },
)
