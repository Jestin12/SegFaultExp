from setuptools import setup
import os
import glob

import os
from glob import glob

package_name = 'drive'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jestin',
    maintainer_email='jestin@example.com',
    description='Drive package',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
			'CoordinateFinder = drive.coordinates:main',
            'driver = drive.driver:main',
        ],
    },
)
