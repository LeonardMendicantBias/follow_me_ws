from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Declare the use_sim_time argument
    package_name = 'follow_me'
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='false', description='Use simulation time'
    )
    
    return LaunchDescription([
        use_sim_time_arg,
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare("nav2_bringup"), "launch", "navigation_launch.py"
                ])
            ),
            launch_arguments={
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                "params_file": PathJoinSubstitution([
                    FindPackageShare(package_name),
                    'config', "nav2_params.yaml"
                ]),
            }.items(),
        )
    ])
