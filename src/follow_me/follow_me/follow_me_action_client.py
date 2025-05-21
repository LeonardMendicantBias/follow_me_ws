import threading

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from follow_me_msgs.action import FollowMe


class FollowMeActionClient(Node):

    def __init__(self):
        super().__init__('follow_me_action_client')
        self._action_client = ActionClient(self, FollowMe, 'follow_me')
        self._goal_handle = None

    def send_goal(self):
        self.get_logger().info(f'Waiting for action server...')
        if self._action_client.wait_for_server(10) is False:
            self.get_logger().info('service not available...')
            return

        goal_msg = FollowMe.Goal()

        self._send_goal_future = self._action_client.send_goal_async(goal_msg)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        self._goal_handle = future.result()
        if not self._goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return

        self.get_logger().info('Goal accepted :)')

        self._get_result_future = self._goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info("FollowMe stops.")

    def cancel_goal(self):
        self.get_logger().info('Canceling goal')
        future = self._goal_handle.cancel_goal_async()
        future.add_done_callback(self.goal_canceled_callback)

    def goal_canceled_callback(self, future):
        cancel_response = future.result()
        if len(cancel_response.goals_canceling) > 0:
            self.get_logger().info('canceling goal success...')
        else:
            self.get_logger().info('canceling goal fail...')

def main(args=None):
    rclpy.init(args=args)

    action_client = FollowMeActionClient()

    def user_input_loop():
        while True:
            user_input = input('\nEnter a "start" for FollowMe, "cancel" to cancel, or "exit" to quit:\n')
            if user_input.lower() == 'exit':
                action_client._running = False
                rclpy.shutdown()
                break
            elif user_input.lower() == 'cancel':
                action_client.cancel_goal()
            elif user_input.lower() == 'start':
                action_client.send_goal()
            else:
                print('Invalid input. Use a positive integer, "cancel", or "exit".')

    input_thread = threading.Thread(target=user_input_loop, daemon=True)
    input_thread.start()

    # rclpy.spin(action_client)
    try:
        rclpy.spin(action_client)
    except KeyboardInterrupt:
        print('\nInterrupted by user.')
        rclpy.shutdown()

    input_thread.join()


if __name__ == '__main__':
    main()