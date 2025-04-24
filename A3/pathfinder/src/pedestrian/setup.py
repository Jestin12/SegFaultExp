from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'pedestrian'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*')),
        (os.path.join('share', package_name, 'Models'), glob('Models/*.pth'))
    ],
    install_requires= [
        'setuptools',
        'sensor_msgs'
        ],
    zip_safe=True,
    maintainer='jestin',
    maintainer_email='jestinji@outlook.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ImageFlipper = pedestrian.ImageFlipper:main',
            'Pedestrian = pedestrian.Pedestrian:main'
        ],
    },
)
