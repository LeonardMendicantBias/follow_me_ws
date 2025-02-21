import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

import threading

from follow_me_msgs.action import FollowMe


class FollowMeActionClient(Node):

	def __init__(self):
		super().__init__('follow_me_action_client')
		self._action_client = ActionClient(self, FollowMe, 'follow_me')
		self.goal_handle = None  # Store the goal handle to cancel later

		# self.declare_parameter('frame_id', 'base_footprint')

	def send_goal(self):
		goal_msg = FollowMe.Goal()
		goal_msg.header.stamp = self.get_clock().now().to_msg()
		goal_msg.header.frame_id = "base_footprint"
		self._action_client.wait_for_server()

		self._goal_future = self._action_client.send_goal_async(goal_msg)

		self._goal_future.add_done_callback(self.goal_response_callback)

	def goal_response_callback(self, future):
		self.goal_handle = future.result()
		if not self.goal_handle.accepted:
			self.get_logger().info('Goal rejected :(')
			return

		self.get_logger().info('Goal accepted :)')

		self._get_result_future = self.goal_handle.get_result_async()
		# self._get_result_future.add_done_callback(self.get_result_callback)

	def cancel_goal(self):
		if self.goal_handle:
			cancel_future = self.goal_handle.cancel_goal_async()
			cancel_future.add_done_callback(self.cancel_response_callback)
			self.goal_handle = None
		else:
			self.get_logger().info('No goal to cancel')

	def cancel_response_callback(self, future):
		cancel_response = future.result()
		if cancel_response.return_code == 0:  # SUCCESS
			self.get_logger().info('Goal successfully canceled')
		else:
			self.get_logger().info('Failed to cancel goal')

def main():
	rclpy.init()
	action_client = FollowMeActionClient()
	
	def user_input_loop():
		while rclpy.ok():
			cmd = input("type start/stop or exit.\n")
			if cmd.lower() == "start":
				action_client.send_goal()
			elif cmd.lower() == "stop":
				action_client.cancel_goal()
			elif cmd.lower() == "exit":
				action_client.get_logger().info('Exiting...')
				break

	user_thread = threading.Thread(target=user_input_loop)
	user_thread.start()
	rclpy.spin(action_client)
	user_thread.join()
	action_client.destroy_node()
	rclpy.shutdown()


if __name__ == '__main__':
	main()