import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'follow_me_action'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Leonard Ngo',
    maintainer_email='ngoak@islab.snu.ac.kr',
    description='Package for Follow Me communication exchange',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'server = follow_me_action.follow_me_action_server:main',
            'client = follow_me_action.follow_me_action_client:main',
        ],
    },
)
