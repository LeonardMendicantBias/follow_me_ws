import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'follow_me_tracking'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'data'), glob(os.path.join(package_name, '*.npy'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Leonard Ngo',
    maintainer_email='ngoak@islab.snu.ac.kr',
    description='ROS package for tracking detected people',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'tracking_service = follow_me_tracking.tracking_service:main',
            'follow_me_server = follow_me_tracking.follow_me_action_server:main',
            'follow_me_client = follow_me_tracking.follow_me_action_client:main',
            'tf2_listener = follow_me_tracking.tf2_listener:main',
        ],
    },
)
