import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'follow_me_robot'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name, 'urdf'), glob(os.path.join('urdf', '*.urdf'))),
        (os.path.join('share', package_name, 'meshes'), glob(os.path.join('models', 'meshes', '*.STL'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='leonard',
    maintainer_email='ngoak@islab.snu.ac.kr',
    description='Package for robot launch files',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        ],
    },
)
