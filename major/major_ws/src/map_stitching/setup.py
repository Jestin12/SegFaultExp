from setuptools import find_packages, setup

package_name = 'map_stitching'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='alfie',
    maintainer_email='alfieeagleton@gmail.com',
    description='Subscribes to camera node, saves the files and stitches them together to make a 2D map.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'map_stitcher = map_stitching.MapStitcher:main',
        ],
    },
)
