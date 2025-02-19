import time

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

from visualization_msgs.msg import Marker, MarkerArray
from upo_laser_people_msgs.msg import PersonDetection, PersonDetectionList

import tf2_geometry_msgs
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped, Pose, Point
from std_msgs.msg import Header

from follow_me_msgs.action import FollowMe
from .kalmanTracker import Tracker


def _quaterion_from_positions(source, dest):# Compute direction vector
	direction = np.array(dest) - np.array(source)
	direction = direction / np.linalg.norm(direction)  # Normalize

	# Define reference direction (assuming forward along X-axis)
	reference = np.array([1.0, 0.0, 0.0])  # Change if needed

	# Compute axis of rotation
	axis = np.cross(reference, direction)
	axis_norm = np.linalg.norm(axis)
	
	if axis_norm < 1e-6:  # If no rotation needed
		return (0.0, 0.0, 0.0, 1.0)

	axis = axis / axis_norm  # Normalize

	# Compute angle using dot product
	angle = np.arccos(np.clip(np.dot(reference, direction), -1.0, 1.0))

	# Convert axis-angle to quaternion
	q = tf_transformations.quaternion_about_axis(angle, axis)
	
	return tuple(q)  # (x, y, z, w)


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

		# an tracking module to associate robot with user
		self.subscription = self.create_subscription(
			PersonDetectionList,
			'detected_people',
			self.detection_callback,
			10
		)
		self.tracker = Tracker(1, 10, 5)

		self.frame_to_track = {}

		# an navigation client to send updated position of following user
		self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

		# publish the updated 
		self.tf_buffer = tf2_ros.Buffer()
		self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

		self.update_publisher = self.create_publisher(PoseStamped, 'goal_update', 10)
		self.publisher_ = self.create_publisher(MarkerArray, 'track_people', 10)
		self.track_colors = [
			(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
			(127, 127, 255), (255, 0, 255), (255, 127, 255),
			(127, 0, 255), (127, 0, 127),(127, 10, 255), (0,255, 127)
		]

	'''
		Update human tracks and send the updated positions to Nav
	'''
	def detection_callback(self, msg: PersonDetectionList):
		detections = np.array([
			[det.position.x, det.position.y]
			for det in msg.detections
		])
		self.tracker.update(detections)

		_msg = MarkerArray()
		for idx, track in enumerate(self.tracker.tracks):
			marker = Marker()
			# marker.header.frame_id = "map"
			marker.header.frame_id = "velodyne"
			# marker.header.stamp = self.get_clock().now().to_msg()
			marker.header.stamp = msg.header.stamp

			marker.id = idx
			marker.type = Marker.CUBE
			marker.action = Marker.ADD

			if len(track.trace) == 0: continue
			marker.pose.position.x = track.trace[-1][0, 0]# + 0.4
			marker.pose.position.y = track.trace[-1][0, 1]# - 0.4

			marker.scale.x = 2.0*0.4
			marker.scale.y = 2.0*0.4
			marker.scale.z = 1.5

			_idx = min(idx, len(self.track_colors)-1)
			marker.color.r = self.track_colors[_idx][0] / 255.
			marker.color.g = self.track_colors[_idx][1] / 255.
			marker.color.b = self.track_colors[_idx][2] / 255.
			marker.color.a = 0.5

			_msg.markers.append(marker)
		self.publisher_.publish(_msg)

	def _get_track_from_frame(self, frame_pos):
		_frame_pos = np.array([
			frame_pos[0], frame_pos[1]
		])
		track_id = None
		min_dist = np.inf
		for track in self.tracker.tracks:
			track_pos = [
				track.trace[-1][0, 0],
				track.trace[-1][0, 1]
			]
			dist = np.linalg.norm(_frame_pos - track_pos)
			if dist < min_dist:
				track_id = track.trackId
				min_dist = dist

		return track_id

	def cancel_callback(self, client):
		frame_id = client.request.header.frame_id
		if frame_id in self.frame_to_track:
			del self.frame_to_track[frame_id]

		self.get_logger().info(f'{frame_id} stops folloing.')
		return CancelResponse.ACCEPT

	def _get_pose(self, track_id, init_pose=None) -> PoseStamped:
		# the latest position of the track is w.r.t. "velodyne"
		_track_position = self.tracker.get_track_position(track_id)
		if _track_position is None: return None

		try:
			t = self.tf_buffer.lookup_transform(
				"map", "velodyne",
				rclpy.time.Time(),
				rclpy.duration.Duration(seconds=5.0)
			)
		except TransformException as ex:
			return None

		# prepare the pose stamped
		track_position = PoseStamped()
		track_position.header.frame_id = "velodyne"
		track_position.header.stamp = self.get_clock().now().to_msg()
		track_position.pose.position.x = _track_position[0]
		track_position.pose.position.y = _track_position[1]
		track_position.pose.position.z = 0.
		track_position.pose.orientation = t.transform.rotation
		# TODO: calculate the direction between the robot and the human
		if init_pose is not None:
			src_pos = np.array([
				track_position.pose.position.x,
				track_position.pose.position.y,
				track_position.pose.position.z,
			])
			tgt_pos = np.array([
				init_pose.pose.position.x,
				init_pose.pose.position.y,
				init_pose.pose.position.z,
			])

			direction = tgt_pos - src_pos
			direction /= np.linalg.norm(direction)
			forward = np.array([1, 0, 0])

			# Compute rotation from forward to direction
			if np.allclose(direction, forward):

				return tuple(src_pos[3:])

			# Compute rotation quaternion
			rot, _ = R.align_vectors([direction], [forward])
			rot = rot.as_quat()
			# apply the facing toward the destination orientation
			track_position.pose.orientation.x = rot[0]
			track_position.pose.orientation.y = rot[1]
			track_position.pose.orientation.z = rot[2]
			track_position.pose.orientation.w = rot[3]

		# then transform to "map" frame
		transformed_pose = tf2_geometry_msgs.do_transform_pose_stamped(
			track_position, t
		)
		return transformed_pose

	def start_callback(self, client: rclpy.action.server.ServerGoalHandle):
		frame_id = client.request.header.frame_id
		if frame_id in self.frame_to_track:
			self.get_logger().info(f'{frame_id} is already following.')
			client.abort()
			return FollowMe.Result()

		# 1/ Get position of robot
		try:
			print("first")
			self.get_logger().info(f"Getting {frame_id}'s transformation.")
			t = self.tf_buffer.lookup_transform(
				"map", frame_id,
				rclpy.time.Time(),
				# 3 sec seems to be the minimum
				# timeout for 5 sec for safe-measure
				rclpy.duration.Duration(seconds=10.0)
			)
			# print(type(t))
		except TransformException as ex:
			# if there is error, reject the following request
			self.get_logger().info(
				f'Could not transformation for {frame_id}: {ex}'
			)
			client.abort()
			return FollowMe.Result()
		
		# The transform from "<frame_id> to map"
		# indicates the position of the robot
		frame_pose = Pose()
		frame_pose.position = Point(
			x=t.transform.translation.x,
			y=t.transform.translation.y,
			z=t.transform.translation.z,
		)
		frame_pose.orientation = t.transform.rotation

		# 2/ indentify the nearest person to follow
		track_id = self._get_track_from_frame(np.array([
			frame_pose.position.x, frame_pose.position.y
		]))
		if track_id is None:
			self.get_logger().info(f"No person to follow.")
			client.abort()
			return FollowMe.Result()
		self.frame_to_track[frame_id] = track_id

		# 3/ initiate follow me
		goal_msg = NavigateToPose.Goal()
		goal_msg.pose.header.frame_id = "map"
		goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
		track_position = self._get_pose(track_id)
		goal_msg.pose = track_position

		self.nav_client.wait_for_server()
		self.send_goal_future = self.nav_client.send_goal_async(goal_msg)
		# _ = await self.send_goal_future 
		# self.send_goal_future.add_done_callback(self.goal_response_callback)

		while rclpy.ok():
			if client.is_cancel_requested:
				client.canceled()
				self.get_logger().info("Goal canceled.")
				return client.Result()
			
			update_goal = PoseStamped()
			update_goal.header.frame_id = "map"
			update_goal.header.stamp = self.get_clock().now().to_msg()

			prev_track_position = track_position
			track_position = self._get_pose(track_id, init_pose=prev_track_position)
			prev_track_position = track_position
			# track_position.header.frame_id = "velodyne"
			# track_position.header.stamp = self.get_clock().now().to_msg()
			# goal_msg.pose = track_position

			self.update_publisher.publish(update_goal)

			time.sleep(0.5)  # sec

		self.get_logger().info(f"Robot with id {frame_id} starts following person with id {track_id}.")
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