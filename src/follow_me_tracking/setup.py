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
            'listener = follow_me_tracking.upo_subscriber:main',
        ],
    },
)
