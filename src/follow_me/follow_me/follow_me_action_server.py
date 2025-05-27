import time

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
import cv2
from cv_bridge import CvBridge
from image_geometry import PinholeCameraModel

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

from filterpy.kalman import KalmanFilter

from follow_me.tracker import Tracker


class FollowMeActionServer(Node):

	def __init__(self):
		super().__init__('follow_me_action_server')
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
			256,
			448
		]
		self.tracker = Tracker(
			img_size=self.img_size,
		)
		self.dt = 1./15  # sec, Hz
		
		self._reset_variable()

		self.marker_publisher = self.create_publisher(
			MarkerArray,
			'/action/marker', 1
		)

	def _reset_variable(self):
		self.request_flag = False
		self.init_flag = False
		self._is_following = False
		self.tracker.reset()
		self.kf = KalmanFilter(dim_x=4, dim_z=2)

		# State transition matrix (x, y, vx, vy)
		self.kf.F = np.array([
			[1, 0, self.dt, 0],
			[0, 1, 0, self.dt],
			[0, 0, 1, 0],
			[0, 0, 0, 1]
		])

		# Measurement function (we observe x and y)
		self.kf.H = np.array([
			[1, 0, 0, 0],
			[0, 1, 0, 0]
		])
		self.kf.P *= 1000.
		self.kf.R *= 0.1
		self.kf.Q = np.eye(4) * 0.01
		self.kf.x = np.zeros((4, 1))

	def _point_to_posestamp(self,
		dest: Point,
		header: Header
	) -> PoseStamped:
		# dest = dest.cpu().numpy()
		dest = Point(x=float(dest[0]), y=float(dest[1]), z=float(dest[2]))
		
		pose = Pose()
		
		# Set position
		pose.position = Point(x=dest.x, y=dest.y, z=dest.z)

		# Create normalized direction vector
		direction = [dest.x, dest.y, dest.z]
		norm = math.sqrt(dest.x**2 + dest.y**2 + dest.z**2)
		if norm == 0:
			raise ValueError("Target position cannot be the origin (0,0,0)")
		direction = [c / norm for c in direction]

		# Define robot forward direction (e.g., x-axis)
		forward = [0.0, 0.0, 1.0]

		# Compute axis of rotation (cross product) and angle (dot product)
		cross = [
			forward[1]*direction[2] - forward[2]*direction[1],
			forward[2]*direction[0] - forward[0]*direction[2],
			forward[0]*direction[1] - forward[1]*direction[0]
		]
		dot = sum(f*d for f, d in zip(forward, direction))
		angle = math.acos(max(min(dot, 1.0), -1.0))  # Clamp dot to [-1, 1]

		# Special case: if vectors are opposite, rotate 180 degrees around Z
		if abs(dot + 1.0) < 1e-6:
			quat = tf_transformations.quaternion_about_axis(math.pi, [0, 0, 1])
		else:
			quat = tf_transformations.quaternion_about_axis(angle, cross)

		# Assign orientation
		pose.orientation = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])

		return PoseStamped(
			header=Header(
				frame_id=header.frame_id,
				stamp=self.get_clock().now().to_msg()
			),
			pose=pose
		)
	
	def _get_target_from_bboxes(self, bboxes, W, H):
		dist = np.array([
			abs(bbox[0]-W) + abs(bbox[1]-H)
			for bbox in bboxes.cpu().numpy()
		])
		return dist.argmax()

	def nav_callback(self, future):
		future.result()
		self._is_following = True

	def yolo_callback(self, msg: ResultArray):
		# if not self.init_flag: return
		_color = ColorRGBA(r=1.0, g=0.8, b=0.1, a=0.5)
		
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
		
		# initiate the FollowMe via sending NavigateToPose
		if not self._is_following:
			H, W, _ = image.shape
			target_id = self._get_target_from_bboxes(bboxes, W, H)
			
			if positions[target_id] is None: return
			# initialize PoseStamped from user's position
			pose_stamp = self._point_to_posestamp(
				positions[target_id],
				msg.image.header
			)
			if pose_stamp is None: return

			goal_msg = NavigateToPose.Goal()
			goal_msg.pose = pose_stamp
			if not self.nav_client.wait_for_server(timeout_sec=1.0):
				self.get_logger().info(f"Navigation server is not online!")
				return FollowMe.Result()
			nav_future = self.nav_client.send_goal_async(goal_msg)
			nav_future.add_done_callback(self.nav_callback)
		else:
			position = None
			if len(bboxes) > 0:
				all_features, visible_part = self.tracker.process_crop(
					image, bboxes, kpts, confs
				)
				target_id = self.tracker.identify(all_features, visible_part)

				# not good enough re-identification
				# if avg_scores[target_id] > 0.5:
				# 	self.get_logger().warn(f"Insufficient re-identification confidence {avg_scores[target_id]}")
				# 	return

				position = positions[target_id]
				if position is not None:
					self.kf.update(np.array([[position[0]], [position[2]]]))

			if position is None:
				self.kf.predict()
				x_pred, z_pred = self.kf.x[0, 0], self.kf.x[1, 0]
				position = [x_pred, 0., z_pred]

			pose_stamp = self._point_to_posestamp(position, msg.image.header)

			pose_stamp = self._point_to_posestamp(
				positions[target_id],
				msg.image.header
			)
			if pose_stamp is None: return
			self.update_publisher.publish(pose_stamp)

		markers = MarkerArray()
		marker = Marker()
		marker.header.frame_id = pose_stamp.header.frame_id
		marker.header.stamp = self.get_clock().now().to_msg()

		marker.ns = 'cylinders'
		marker.id = 0
		marker.type = Marker.CUBE
		marker.action = Marker.ADD
		marker.pose.position = pose_stamp.pose.position
		marker.pose.orientation = pose_stamp.pose.orientation
		marker.scale.x = 0.2 # diameter in x
		marker.scale.y = 0.2  # diameter in y
		marker.scale.z = 1.0  
		marker.lifetime.sec = 0  # 0 means forever

		marker.color = _color
		markers.markers.append(marker)
		self.marker_publisher.publish(markers)

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