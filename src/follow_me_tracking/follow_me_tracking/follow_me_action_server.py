import time

import rclpy
from rclpy.action import ActionClient
from rclpy.action import ActionServer
from rclpy.node import Node

from action_tutorials_interfaces.action import Fibonacci
from upo_laser_people_msgs.msg import PersonDetection, PersonDetectionList

from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped, Pose

from fmp_msgs.msg import FollowMe
from kalmanTracker import Tracker


class FollowMePyServer(Node):

	def __init__(self):
		super().__init__('follow_me_action_server')
		
		# an action server to receive request from robot/user
		self._action_server = ActionServer(
			self,
			FollowMe,
			'follow_me',
			self.recieve_goal_callback
		)

		# an tracking module to associate robot with user
		self.subscription = self.create_subscription(
			PersonDetectionList,
			'detections',
			self.detection_callback,
			10
		)
		self.tracker = Tracker(150, 30, 5)

		self.frame_to_track = {}

		# an navigation client to send updated position of following user
		self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

		# publish the updated 
		self.update_publisher = self.create_publisher(PoseStamped, 'goal_update', 10)

	def detection_callback(self, msg: PersonDetectionList):
		# frame_id = msg.header.frame_id
		detections = [
			[det.position.x, det.position.y]
			for det in msg.detections
		]
		self.tracker.update(detections)

		# # send update positions to nav_client
		# for frame_id, track_id in self.frame_to_track.items():
		# 	position = self.tracker.get_position_track(track_id)
		# 	pose_stamp = PoseStamped()
		# 	pose_stamp.header.time = self.get_clock().now().to_msg()
		# 	pose_stamp.header.frame_id = frame_id

		# 	pose_msg = Pose()
		# 	pose_msg.position.x = position.x
		# 	pose_msg.position.y = 0
		# 	pose_msg.position.z = position.y

		# 	pose_msg.orientation.x = 0.0
		# 	pose_msg.orientation.y = 0.0
		# 	pose_msg.orientation.z = 0.0
		# 	pose_msg.orientation.w = 1.0 
		# 	pass
		

	def recieve_goal_callback(self, goal_handle):
		frame_id = goal_handle.request.frame_id
		# if frame_id in list(self.frame_to_track.keys()):
		#	# reject request
		# 	break

		# Identify the closest detected person to the frame_id
		track_id = 0
		self.frame_to_track[frame_id] = track_id
		self.get_logger().info(f"{frame_id} starts following {track_id}")

		position = self.tracker.get_position_track(track_id)
		goal_msg = NavigateToPose.Goal()
		goal_msg.pose.header.frame_id = "map"
		goal_msg.pose.header.stamp = self.get_clock().now().to_msg()

		# Generate random coordinates within a predefined range
		goal_msg.pose.pose.position.x = position.x
		goal_msg.pose.pose.position.y = position.y
		goal_msg.pose.pose.orientation.w = 1.0  # No rotation

		self.nav_client.wait_for_server()
		self.send_goal_future = self.action_client.send_goal_async(goal_msg)
		# self.send_goal_future.add_done_callback(self.goal_response_callback)

		# goal_handle.succeed()

		result = FollowMe.Result()
		return result


def main(args=None):
	rclpy.init(args=args)

	fibonacci_action_server = FollowMePyServer()

	rclpy.spin(fibonacci_action_server)


if __name__ == '__main__':
	main()