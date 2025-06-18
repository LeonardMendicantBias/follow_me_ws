import time
from typing import List
import rclpy
from rclpy.action import ActionServer, ActionClient, CancelResponse, GoalResponse
# from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener, TransformException
import numpy as np
import math

import tf_transformations
# from tf2_ros.transformations import do_transform_pose
import tf2_geometry_msgs
from tf2_geometry_msgs import do_transform_pose
import cv2
from cv_bridge import CvBridge
from image_geometry import PinholeCameraModel

from tf2_geometry_msgs import do_transform_pose as tf2_do_transform_pose
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion, Pose2D
from vision_msgs.msg import BoundingBox2D
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA

from follow_me_msgs.msg import ResultArray, YoloResult
from follow_me_msgs.action import FollowMe

import torch
import torch.nn.functional as F
from torch import nn

from torchvision.models import resnet18 as resnet, ResNet18_Weights
from torchvision import transforms

from filterpy.kalman import ExtendedKalmanFilter, KalmanFilter

from follow_me.tracker import Tracker


class FollowMeActionServer(Node):

	def __init__(self):
		super().__init__('follow_me_action_server')
		self.declare_parameter('img_width', 448)
		self.declare_parameter('img_height', 256)

		self.tf_buffer = Buffer()
		self.tf_listener = TransformListener(self.tf_buffer, self)
	
		self.cv_bridge = CvBridge()

		self._action_server = ActionServer(self,
			FollowMe,
			'follow_me',
			goal_callback=self.goal_callback,
			execute_callback=self.execute_callback,
			cancel_callback=self.cancel_callback
		)

		self.cam_model = None  # PinholeCameraModel()
		# self.info_sub = self.create_subscription(
		# 	CameraInfo,
		# 	'/camera/camera/color/camera_info',
		# 	self.info_callback, 1
		# )
		self.sub = self.create_subscription(
			ResultArray,
			'/yolo',
			self.yolo_callback, 1
		)
		self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
		self.update_publisher = self.create_publisher(PoseStamped, 'goal_update', 1)
		
		self.img_size = [
			self.get_parameter('img_height').get_parameter_value().integer_value,
			self.get_parameter('img_width').get_parameter_value().integer_value
		]
		self.tracker = Tracker(
			img_size=self.img_size,
		)
		self.dt = 1./15  # sec, Hz
		
		self._reset_variable()

		self.pose_publisher = self.create_publisher(
			PoseStamped,
			'/action/posestamp', 1
		)
		self.kalman_publisher = self.create_publisher(
			PoseStamped,
			'/action/kalman', 1
		)

	def _reset_variable(self):
		self._is_dirty = False
		self.request_flag = False
		self.init_flag = False
		self._is_following = False
		self.tracker.reset()
		self._last_poses: List[Pose] = []
		self.kf = KalmanFilter(dim_x=4, dim_z=2)

		self.kf.x = np.array([0, 0, 0, 0])

		# Covariances
		self.kf.F = np.array([
			[1, 0, self.dt, 0],
			[0, 1, 0, self.dt],
			[0, 0, 1,  0],
			[0, 0, 0,  1]
		])
		self.kf.H = np.array([
			[1, 0, 0, 0],
			[0, 1, 0, 0]
		])
		self.kf.P *= 500.
		self.kf.R *= 5.

		self._last_posestamped: PoseStamped = None

	def _position_to_pose(self, dest: Point) -> Pose:
		dest = Point(x=float(dest[0]), y=0., z=float(dest[2]))
		
		yaw = math.atan2(dest.z, dest.x)
		q = tf_transformations.quaternion_from_euler(0, -yaw, 0)
		return Pose(
			position=dest,
			orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
		)
	
	def _get_target_from_bboxes(self, bboxes, W, H):
		dist = np.array([
			abs(bbox[0]-W/2) + abs(bbox[1]-H/2)
			for bbox in bboxes
		])
		return dist.argmin()

	def nav_callback(self, future):
		future.result()
		self.get_logger().info(f"first destination")
		self._is_following = True

	def _get_init_pose(self, image, bboxes, positions) -> Pose:
		H, W, _ = image.shape
		dist = np.array([
			abs(bbox[0]-W/2) + abs(bbox[1]-H/2)
			for bbox in bboxes
		])
		target_id = dist.argmin()
		
		if (position := positions[target_id]) is None: return

		# self.kf.x = np.array([position[0], position[2], 0, 0])
		return target_id, self._position_to_pose(position)

	def _reidentify(self, image, bboxes, kpts, confs, positions):
		all_features, visible_part = self.tracker.process_crop(
			image, bboxes, kpts, confs
		)
		target_id = self.tracker.identify(
			all_features.cpu().detach(),
			visible_part.cpu().detach(),
		)
		if target_id is not None: 
			self.tracker.update(
				target_id, 
				all_features.cpu().detach(),
				visible_part.cpu().detach()
			)
			position = positions[target_id]
			if position is not None:
				# z = np.array([position[0], position[2]])
				# self.kf.update(z)
				return self._position_to_pose(position)

	def _extrapolate(self) -> Pose:
		x_pos, y_pos, v_x, v_y = self.kf.x
		dest = Point(x=float(x_pos), y=float(y_pos), z=0.)
		# dest = Point(x=float(x_pos), y=0., z=float(y_pos))
		_last_pose = self._last_poses[-min(3, len(self._last_poses))]

		yaw = math.atan2(
			# dest.z - _last_pose.position.z,
			dest.y - _last_pose.position.y,
			dest.x - _last_pose.position.x,
		)
		# q = tf_transformations.quaternion_from_euler(0, -yaw, 0)
		q = tf_transformations.quaternion_from_euler(0, 0, yaw)
		return Pose(
			position=Point(x=float(x_pos), y=float(y_pos), z=0.),
			orientation=Quaternion(x=q[0], y=q[1], z=q[2], w=q[3]),
			# orientation=Quaternion(x=0., y=0., z=0., w=q[3]),
		)

	def _update_last_pose(self, pose):
		if len(self._last_poses) == 0:
			self._last_poses.append(pose)
		else:
			_last_pose = self._last_poses[-1]
			# print(_last_pose)
			dist = np.linalg.norm([
				_last_pose.position.x - pose.position.x,
				# _last_pose.position.z - pose.position.z,
				_last_pose.position.y - pose.position.y,
			])
			if dist > 0.025:
				self._last_poses.append(pose)
		if len(self._last_poses) > 5:
			self._last_poses.pop(0)

	def yolo_callback(self, msg: ResultArray):
		image = self.cv_bridge.imgmsg_to_cv2(msg.image, "bgr8")

		bboxes = [
			[
				ret.bbox.center.position.x, ret.bbox.center.position.y,
				ret.bbox.size_x, ret.bbox.size_y
			] for ret in msg.results
		]
		kpts = [
			[
				[kpt.x, kpt.y]
				for kpt in ret.kpts
			]
			for ret in msg.results
		]
		confs = [
			ret.confidences
			for ret in msg.results
		]
		positions = [
			[ret.position.x, ret.position.y, ret.position.z]
			for ret in msg.results
		]
		_is_human = (np.array(confs) > 0.5).any()
		
		if not self.init_flag: return

		# initiate the FollowMe via sending NavigateToPose
		self.kf.predict()

		_header = Header(
			frame_id=msg.image.header.frame_id,
			stamp=self.get_clock().now().to_msg()
		)
		if not self._is_following:
			if len(bboxes) == 0: return

			target_id, pose = self._get_init_pose(image, bboxes, positions)
			if pose is None: return
			
			# init kalman filter
			try:
				transform = self.tf_buffer.lookup_transform(
					"map",
					msg.image.header.frame_id,
					rclpy.time.Time(),
					timeout=rclpy.duration.Duration(seconds=0.5)
				)
				t_pose = tf2_do_transform_pose(pose, transform)
				self.kf.x = np.array([t_pose.position.x, t_pose.position.y, 0, 0])
				self._update_last_pose(t_pose)
			except Exception as e:
				self.get_logger().warn(f"Cannot initialize Kalman filter due to transformation: {e}")
				return

			goal_msg = NavigateToPose.Goal()
			goal_msg.pose = PoseStamped(header=_header, pose=pose)
			
			if not self.nav_client.wait_for_server(timeout_sec=1.0):
				self.get_logger().info(f"Navigation server is not online!")
				return FollowMe.Result()
			nav_future = self.nav_client.send_goal_async(goal_msg)
			nav_future.add_done_callback(self.nav_callback)
			
			all_features, visible_part = self.tracker.process_crop(
				image, bboxes, kpts, confs
			)
			self.tracker.update(
				target_id, 
				all_features.cpu().detach(),
				visible_part.cpu().detach()
			)

			self._is_following = True
		else:  # Either extract new location from bboxes
			_header.frame_id = "map"
			pose = self._reidentify(
				image, bboxes, kpts, confs, positions
			) if _is_human else None
			# if _is_human and self._is_dirty: self._is_dirty = False
			
			_flag = False
			if pose is None:
				pose = self._extrapolate()
				_flag = True
			else:
				# update kalman filter
				self._is_dirty = False
				try:
					transform = self.tf_buffer.lookup_transform(
						"map",
						msg.image.header.frame_id,
						rclpy.time.Time(),
						timeout=rclpy.duration.Duration(seconds=0.2)
					)
					pose = tf2_do_transform_pose(pose, transform)
					self.kf.update([pose.position.x, pose.position.y])
					self._update_last_pose(pose)
				except Exception as e:
					self.get_logger().warn(f"Cannot update Kalman filter due to transformation: {e}")
					return

			pose_stamped = PoseStamped(header=_header, pose=pose)
			if not self._is_dirty:
				self.update_publisher.publish(pose_stamped)
				self.pose_publisher.publish(pose_stamped)
			if _flag: self._is_dirty = True
			# if not _is_human: self._is_dirty = True
			# else: self._is_dirty = False
			
			# Kalman debugging

			_header.frame_id = "map"
			self.kalman_publisher.publish(PoseStamped(header=_header, pose=self._extrapolate()))

		# all_features, visible_part = self.tracker.process_crop(
		# 	image, bboxes, kpts, confs
		# ) if _is_human else None, None
		# if _is_human:
		# 	target_id = self.tracker.identify(
		# 		all_features.cpu().detach(),
		# 		visible_part.cpu().detach(),
		# 	) if self._is_following else self._get_init_pose(image, bboxes, positions)
		# 	if target_id is None:
		# 		return
		# 	position = positions[target_id]
		# else:
		# 	position = positions[target_id]

	def goal_callback(self, goal_request):
		self.init_flag = True
		"""Accept or reject a client request to begin an action."""
		# This server allows multiple goals in parallel
		self.get_logger().info('Received goal request')
		return GoalResponse.ACCEPT

	def execute_callback(self, goal_handle):
		self.get_logger().info('Executing goal...')

		while rclpy.ok():
			if not goal_handle.is_active:
				self.get_logger().info('Goal aborted')
				break

			if goal_handle.is_cancel_requested:
				goal_handle.canceled()
				self.get_logger().info('Goal canceled')
				break

			rclpy.spin_once(self, timeout_sec=1.0)

		# goal_handle.succeed()
		return FollowMe.Result()

	def cancel_callback(self, goal_handle):
		self._reset_variable()
		"""Accept or reject a client request to cancel an action."""
		self.get_logger().info('Received cancel request')
		return CancelResponse.ACCEPT


def main(args=None):
	rclpy.init(args=args)

	action_server = FollowMeActionServer()

	rclpy.spin(action_server)


if __name__ == '__main__':
	main()