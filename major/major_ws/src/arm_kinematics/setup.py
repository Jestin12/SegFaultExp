from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'arm_kinematics'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        (os.path.join('share', package_name, 'launch'),  glob('launch/*.py')), # Include config files 
        (os.path.join('share', package_name, 'config'),  glob('config/*.yaml')),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jestin',
    maintainer_email='aarc0926@uni.sydney.edu.au',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
			'arm_kinematics = arm_kinematics.kinematics:main',
            'servo = arm_kinematics.servo:main',
            'tester = arm_kinematics.tester:main'
        ],
    },
)
