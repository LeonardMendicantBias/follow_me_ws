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

from scipy.spatial.transform import Rotation as R

from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion
from std_msgs.msg import Header

from follow_me_msgs.srv import PersonId, PersonTracking
from follow_me_msgs.action import FollowMe


class FollowMePyServer(Node):

	def __init__(self):
		super().__init__('follow_me_action_server')
		
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

		self.declare_parameter('update_freq', 5)

	def cancel_callback(self, client):
		frame_id = client.request.header.frame_id
		if frame_id in self.frame_to_track:
			del self.frame_to_track[frame_id]

		self.get_logger().info(f'{frame_id} stops following.')
		return CancelResponse.ACCEPT

	def _calculate_pose(self, source: Point, dest: Point) -> Pose:
		# direction from source toward dest
		yaw = math.atan2(source.y - dest.y, source.x - dest.x)
		q = tf_transformations.quaternion_from_euler(0, 0, yaw)

		return Pose(
			position=dest,
			orientation=Quaternion(*q)
		)

	def _get_position(self, frame_id, stamp=None) -> Point:
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
		
		return Point(
			x=t.transform.translation.x,
			y=t.transform.translation.y,
			z=t.transform.translation.z,
		)

	def _get_track_position(self, track_id) -> Point:
		tracking_req = PersonTracking.Request(track_id=track_id)
		future = self.tracking_service_client.call_async(tracking_req)
		rclpy.spin_until_future_complete(self, future)
		tracking_response = future.result()

		return tracking_response.position

	'''
		Processing "Follow me" request
		1/ Get id (frame_id) and position of requesting robot
		2/ Identify person_id via personId service
		3/ Initiate following 
	'''
	def start_callback(self, client: rclpy.action.server.ServerGoalHandle):
		_update_freq = self.get_parameter('update_freq').get_parameter_value().integer_value
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

		# request the position of track_id w.r.t. "map"
		track_position: Point = self._get_track_position(track_id)
		
		self.get_logger().info(f"{frame_id} initiates follow me procedure.")
		prev_robot_position: Point = self._get_position(frame_id) # client.request.header.stamp)
		if prev_robot_position is None:
			client.abort()
			return FollowMe.Result()
		
		_pose: Pose = self._calculate_pose(prev_robot_position, track_position)
		goal_msg.pose = PoseStamped(
			header=Header(
				frame_id="map",
				stamp=self.get_clock().now().to_msg()
			),
			pose=_pose
		)

		if not self.nav_client.wait_for_server(timeout_sec=1.0):
			self.get_logger().info(f"Navigation server is not online!")
			client.abort()
			return FollowMe.Result()
		send_goal_future = self.nav_client.send_goal_async(goal_msg)

		###
		# rclpy.spin_until_future_complete(self, send_goal_future)
		# self.send_goal_future.add_done_callback(self.goal_response_callback)
		###

		self.get_logger().info(f"Robot with id {frame_id} starts following person with id {track_id}.")
		while rclpy.ok():
			if client.is_cancel_requested:
				client.canceled()
				self.get_logger().info("Goal canceled.")
				# del self.frame_to_track[frame_id]
				return client.Result()
			

			_cur_position: Point = self._get_position(frame_id)
			if _cur_position is None:
				time.sleep(0.1)
				continue

			track_position: Point = self._get_track_position(track_id)
			if track_position is None:
				self.get_logger().info("Cannot get new position")
				time.sleep(0.1)
				continue

			update_goal = PoseStamped(
				header=Header(
					frame_id="map",
					stamp=self.get_clock().now().to_msg()
				),
				pose=self._calculate_pose(_cur_position, track_position)
			)
			self.update_publisher.publish(update_goal)

			time.sleep(1/_update_freq)  # sec
		
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