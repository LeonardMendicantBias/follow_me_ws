import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose

from fmp_msgs.msg import FollowMe


class FollowMePyClient(Node):

    def __init__(self):
        super().__init__('follow_me_action_client')
        self._action_client = ActionClient(self,
            FollowMe,
            'follow_me'
        )

    def send_goal(self, frame_id):
        goal_msg = FollowMe.Goal()
        goal_msg.frame_id = frame_id

        self._action_client.wait_for_server()

        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            # feedback_callback=self.feedback_callback
        )

        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Follow rejected')
            return

        self.get_logger().info('Following')

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        future.result()
        self.get_logger().info('Finished following')

    # def feedback_callback(self, feedback_msg):
    #     feedback = feedback_msg.feedback
    #     self.get_logger().info('Received feedback: {0}'.format(feedback.partial_sequence))


def main():
    rclpy.init()

    minimal_client = FollowMePyClient()
    try:
        while True:
            if input('Enter "send" for sending command'):
                future = minimal_client.send_goal("base_footprint")
                rclpy.spin_until_future_complete(minimal_client, future)
                response = future.result()
            
    except KeyboardInterrupt:
        print('interrupted!')

    minimal_client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()