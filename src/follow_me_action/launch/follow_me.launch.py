import launch
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    """Generate launch description with multiple components."""
    container = ComposableNodeContainer(
        name='follow_me_container',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[
            ComposableNode(
                package='follow_me_action',
                name='follow_me_action_server',
            ),
            ComposableNode(
                package='follow_me_action',
                name='follow_me_action_client',
            )
        ],
        output='both',
    )

    return launch.LaunchDescription([container])