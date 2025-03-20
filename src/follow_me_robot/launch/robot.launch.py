import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch_ros.actions import Node
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution, LaunchConfiguration
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster


def generate_launch_description():
    # TURTLEBOT3_MODEL = os.environ["TURTLEBOT3_MODEL"]
    ROBOT_MODEL = "tracer_mini"

    package_name = "follow_me_py"  # Replace with your package name
    robot_description_file = "tracer_mini.urdf"  # Replace with your URDF file name
    controller_config_file = "diff_drive_controller.yaml"  # Controller configuration file

    # package_share_directory = FindPackageShare(package=package_name).find(package_name)
    package_share_directory = get_package_share_directory(package_name)
    # robot_description_path = os.path.join(package_share_directory, "urdf", LaunchConfiguration('robot_description_file'))
    controller_config_path = os.path.join(package_share_directory, "urdf", controller_config_file)

    return LaunchDescription([
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="false",
            description="Use simulation (Gazebo) clock if true"
        ),
        DeclareLaunchArgument(
            'port_name',
            default_value='can1',
            description='CAN bus name, e.g. can1'
        ),
        DeclareLaunchArgument(
            'odom_frame',
            default_value='odom',
            description='Odometry frame id'
        ),
        DeclareLaunchArgument(
            'base_frame',
            # default_value='base_link',
            default_value='base_footprint',
            description='Base link frame id'
        ),
        DeclareLaunchArgument(
            'odom_topic_name',
            default_value='odom',
            description='Odometry topic name'
        ),
        DeclareLaunchArgument(
            'control_rate', 
            default_value='50',
            description='Simulation control loop update rate'
        ),
        DeclareLaunchArgument(
            'robot_description_file', 
            default_value='tracer_mini.urdf',
            description='URDF file for the robot'
        ),

        Node(
            package='tracer_base',
            executable='tracer_base_node',
            output='screen',
            emulate_tty=True,
            parameters=[{
                'is_tracer_mini': True,
                'simulated_robot': False,
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'port_name': LaunchConfiguration('port_name'),                
                'odom_frame': LaunchConfiguration('odom_frame'),
                'base_frame': LaunchConfiguration('base_frame'),
                'odom_topic_name': LaunchConfiguration('odom_topic_name'),
                'control_rate': LaunchConfiguration('control_rate'),
            }]
        ),
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            output="screen",
            parameters=[{
                "use_sim_time": LaunchConfiguration('use_sim_time'),
                "robot_description": ParameterValue(Command(['xacro ', PathJoinSubstitution([
                    FindPackageShare(package_name),
                    'urdf',
                    LaunchConfiguration('robot_description_file')
                ])]), value_type=str)
            }],
        ),
        Node(
            package='joint_state_publisher',
            executable='joint_state_publisher',
            arguments=[
                PathJoinSubstitution([
                    FindPackageShare('follow_me'),
                    'urdf',
                    LaunchConfiguration('robot_description_file')
                ])
            ]
        ),
    ])