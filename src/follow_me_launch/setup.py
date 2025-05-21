import glob
from setuptools import find_packages, setup

package_name = 'follow_me_launch'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob.glob('launch/*.launch')),
        ('share/' + package_name + '/urdf', glob.glob('urdf/*.urdf')),
        ('share/' + package_name + '/meshes', glob.glob('meshes/*.STL')),
        ('share/' + package_name + '/config', glob.glob('config/*.xml')),
        ('share/' + package_name + '/config', glob.glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nardy',
    maintainer_email='nardy@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        ],
    },
)
