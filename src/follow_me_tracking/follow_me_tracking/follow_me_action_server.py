import time
import math

import rclpy
from rclpy.action import ActionServer, ActionClient, CancelResponse
import rclpy.action
from rclpy.node import Node

import rclpy.time
import tf_transformations
import tf2_ros
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

import numpy as np
from scipy.spatial.transform import Rotation as R

import tf2_geometry_msgs
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped, Pose, Point
from std_msgs.msg import Header

from follow_me_msgs.srv import PersonId, PersonTracking
from follow_me_msgs.action import FollowMe


class FollowMePyServer(Node):

	def __init__(self):
		super().__init__('follow_me_action_server')
		# self.target_frame = self.declare_parameter('target_frame', 'base_footprint').get_parameter_value()
		
		# an action server to receive request from robot/user
		self._action_server = ActionServer(
			self,
			FollowMe,
			'follow_me',
			execute_callback=self.start_callback,
			cancel_callback=self.cancel_callback
		)
		self.id_service_client = self.create_client(PersonId, 'person_id')
		while not self.id_service_client.wait_for_service(timeout_sec=1.0):
			self.get_logger().info('id service not available, waiting again...')
			
		self.tracking_service_client = self.create_client(
			PersonTracking, 'person_tracking'
		)
		while not self.tracking_service_client.wait_for_service(timeout_sec=1.0):
			self.get_logger().info('tracking service not available, waiting again...')

		self.frame_to_track = {}

		# an navigation client to send updated position of following user
		self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

		# publish the updated 
		self.tf_buffer = tf2_ros.Buffer()
		self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

		self.update_publisher = self.create_publisher(PoseStamped, 'goal_update', 10)

	def cancel_callback(self, client):
		frame_id = client.request.header.frame_id
		if frame_id in self.frame_to_track:
			del self.frame_to_track[frame_id]

		self.get_logger().info(f'{frame_id} stops following.')
		return CancelResponse.ACCEPT

	def _fix_posestamp(self, source: PoseStamped, dest: PoseStamped) -> PoseStamped:
		print(source.header.frame_id, dest.header.frame_id)
		# if source.header.frame_id != dest.header.frame_id:
		# 	print("not in the same frame")
		# 	return None
		
		x1, y1 = source.pose.position.x, source.pose.position.y
		x2, y2 = dest.pose.position.x, dest.pose.position.y

		# direction source -> dest
		yaw = math.atan2(y1 - y2, x1 - x2)
		quaternion = tf_transformations.quaternion_from_euler(0, 0, yaw)

		dest.pose.orientation.x = quaternion[0]
		dest.pose.orientation.y = quaternion[1]
		dest.pose.orientation.z = quaternion[2]
		dest.pose.orientation.w = quaternion[3]

		return dest

	def _get_position(self, frame_id, stamp=None) -> PoseStamped:
		try:
			self.get_logger().info(f"Getting {frame_id}'s transformation.")
			t = self.tf_buffer.lookup_transform(
				"map", frame_id,
				stamp if stamp is not None else rclpy.time.Time(),
				# 3 sec seems to be the minimum
				# timeout for 5 sec for safe-measure
				rclpy.duration.Duration(seconds=5.0)
			)
		except TransformException as ex:
			# if there is error, reject the following request
			self.get_logger().info(
				f'Could not transformation for {frame_id}: {ex}'
			)
			return None
		
		# The position of robot w.r.t. frame "map"
		posestamp = PoseStamped()
		posestamp.header.stamp = self.get_clock().now().to_msg()
		posestamp.header.frame_id = "map"  # client.request.header
		posestamp.pose.position = Point(
			x=t.transform.translation.x,
			y=t.transform.translation.y,
			z=t.transform.translation.z,
		)
		posestamp.pose.orientation = t.transform.rotation
		return posestamp

	'''
		Processing "Follow me" request
		1/ Get id (frame_id) and position of requesting robot
		2/ Identify person_id via personId service
		3/ Initiate following 
	'''
	def start_callback(self, client: rclpy.action.server.ServerGoalHandle):
		frame_id = client.request.header.frame_id

		# if frame_id in self.frame_to_track:
		# 	self.get_logger().info(f'{frame_id} is already following.')
		# 	client.abort()
		# 	return FollowMe.Result()

		# 2/ indentify the nearest person to follow
		id_req = PersonId.Request(header=client.request.header)
		future = self.id_service_client.call_async(id_req)
		rclpy.spin_until_future_complete(self, future)
		id_response = future.result()
		track_id = id_response.track_id

		if track_id is None:
			self.get_logger().info(f"No person to follow.")
			client.abort()
			return FollowMe.Result()
		else:
			print("track_id", track_id)
		# self.frame_to_track[frame_id] = track_id

		# 3/ initiate follow me
		goal_msg = NavigateToPose.Goal()

		tracking_req = PersonTracking.Request(track_id=track_id)
		future = self.tracking_service_client.call_async(tracking_req)
		rclpy.spin_until_future_complete(self, future)
		tracking_response = future.result()
		track_posestamp = tracking_response.pose  # w.r.t. frame "map"
		
		# self.get_logger().info(f"{frame_id} initiates follow me procedure.")
		print(f"{frame_id} initiates follow me procedure.")
		prev_robot_posestamp = self._get_position(frame_id) # client.request.header.stamp)
		if prev_robot_posestamp is None:
			client.abort()
			return FollowMe.Result()
		
		_posestamp = self._fix_posestamp(prev_robot_posestamp, track_posestamp)
		goal_msg.pose = _posestamp

		self.nav_client.wait_for_server()
		send_goal_future = self.nav_client.send_goal_async(goal_msg)

		###
		# rclpy.spin_until_future_complete(self, send_goal_future)
		# self.send_goal_future.add_done_callback(self.goal_response_callback)
		###

		self.get_logger().info(f"Robot with id {frame_id} starts following person with id {track_id}.")
		counter = 0
		while rclpy.ok():
			if client.is_cancel_requested:
				client.canceled()
				self.get_logger().info("Goal canceled.")
				return client.Result()
			
			tracking_req = PersonTracking.Request(track_id=track_id)
			future = self.tracking_service_client.call_async(tracking_req)
			rclpy.spin_until_future_complete(self, future)
			tracking_response = future.result()
			track_posestamp = tracking_response.pose
			if track_posestamp is None:
				print("can't get new position")
				continue

			update_goal = PoseStamped()
			update_goal.header.frame_id = "map"
			update_goal.header.stamp = self.get_clock().now().to_msg()
			# update_goal.pose = track_posestamp.pose

			_posestamp = self._get_position(frame_id)
			if _posestamp is None:
				continue
			next_posestamp = self._fix_posestamp(_posestamp, track_posestamp)
			if _posestamp is None:
				print("Can't fix pose")
				continue
			update_goal.pose = next_posestamp.pose
			self.update_publisher.publish(update_goal)
			# _posestamp = track_posestamp

			print(f"\r{counter} new destination", track_posestamp, end="")
			counter += 1

			time.sleep(0.2)  # sec
		
		client.succeed()
		result = FollowMe.Result()
		return result


def main(args=None):
	rclpy.init(args=args)

	action_server = FollowMePyServer()

	rclpy.spin(action_server)
	# rclpy.shutdown()


if __name__ == '__main__':
	main()