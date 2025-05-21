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

from sklearn.linear_model import Ridge


class FollowMeActionServer(Node):

	def __init__(self):
		super().__init__('follow_me_action_server')
		self.tf_buffer = Buffer()
		self.tf_listener = TransformListener(self.tf_buffer, self)
	
		self.cv_bridge = CvBridge()

		self.is_following = False
		self.last_user_position = None
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
		
		self.img_size = [256, 448]
		self.conf_thres = 0.5
		self.part_ids = [1, 5, 6, 11, 12, 13, 14, 15, 16]
		self.part_masks = torch.tensor([
			[1, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 1, 1, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 1, 1, 1, 1, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 1, 1],
		], device="cuda")
		self.N = self.part_masks.shape[0]

		self.memory_size = 10
		self.min_pos_example = 1
		
		model = resnet(weights=ResNet18_Weights.IMAGENET1K_V1)
		self.feat_extractor = nn.Sequential(*list(model.children())[:-2])
		self.feat_extractor.eval()
		self.feat_extractor.cuda()

		self.preprocess = transforms.Compose([
			transforms.ToPILImage(),
			transforms.Resize((256, 128)),
			transforms.ToTensor(),
			transforms.Normalize(
				mean=[0.485, 0.456, 0.406],
				std=[0.229, 0.224, 0.225]
			)
		])

		self._reset_variable()

		self.marker_publisher = self.create_publisher(
			MarkerArray,
			'/action/marker', 1
		)

	def _get_resnet(self):
		preprocess = transforms.Compose([
			transforms.ToPILImage(),
			transforms.Resize((256, 128)),
			transforms.ToTensor(),
			transforms.Normalize(
				mean=[0.485, 0.456, 0.406],
				std=[0.229, 0.224, 0.225]
			)
		])
		
		model = resnet(weights=ResNet18_Weights.IMAGENET1K_V1)
		feat_extractor = nn.Sequential(*list(model.children())[:-2])
		feat_extractor.eval()
		feat_extractor.cuda()

		return preprocess, feat_extractor

	def _reset_variable(self):
		self.request_flag = False
		self.init_flag = False
		self.last_user_position = None
		self.is_following = False
		self.lt_pos_memory = [[] for _ in range(self.N+1)]
		self.st_pos_memory = [[] for _ in range(self.N+1)]
		self.lt_neg_memory = [[] for _ in range(self.N+1)]
		self.st_neg_memory = [[] for _ in range(self.N+1)]
		self.classifiers = [Ridge(alpha=1.0) for _ in range(self.N+1)]

	# def info_callback(self, msg: CameraInfo):
	# 	if isinstance(self.cam_model, PinholeCameraModel): return

	# 	self.cam_model = PinholeCameraModel()
	# 	self.cam_model.fromCameraInfo(msg)

	def normalize(self, v):
		norm = np.linalg.norm(v)
		return v / norm if norm > 0 else v

	# pose_stamp = self._point_to_posestamp(positions[target_id], msg.image.header)
	def _point_to_posestamp(self,
		dest: Point,
		header: Header
	) -> PoseStamped:
		dest = dest.cpu().numpy()
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

	def _process_crops(self,
		bboxes,
		kpts,
		confs,
		crop_imgs
	):
		bbox_features = self.feat_extractor(
			torch.stack([
				self.preprocess(crop_img) for crop_img in crop_imgs
			]).cuda()
		)
		K, N, D = kpts.shape[0], self.N, bbox_features.shape[-1]

		visible_kpt_indicator = (confs[:, self.part_ids].unsqueeze(1) * self.part_masks.unsqueeze(0)) > self.conf_thres
		visible_part_indicator = visible_kpt_indicator.amax(-1)
		# (K, N, len(self.part_ids)), (K, N)

		part_kpts = kpts[:, self.part_ids]  # (K, N, 2): K people, N parts, 2D coordinates
		y = part_kpts[:, :, 1].unsqueeze(1) * self.part_masks.unsqueeze(0)  # select the corresponding parts
		y = y * visible_kpt_indicator  # ignore the invisible parts
		
		#
		y_min = y.clone()
		y_min[y_min == 0] = torch.inf
		y_min = y_min.amin(dim=-1)
		y_min[y_min == torch.inf] = 0
		y_min[:, 0] = bboxes[:, 1]

		y_max = y.amax(dim=-1)
		y_max = torch.concat([y_max, bboxes[:, 3:4]], dim=1)[:, 1:]

		# 
		patch_pixel_size = torch.div(bboxes[:, 3] - bboxes[:, 1], bbox_features.shape[-2])
		coor = torch.arange(bbox_features.shape[-2], device=bbox_features.device).unsqueeze(0).unsqueeze(0)  # (1, 1, H)
		coor = coor.expand(K, N, -1)

		y_min_stt = torch.div(y_min - bboxes[:, 1:2], patch_pixel_size[..., None])
		min_coor = torch.ge(coor, y_min_stt.long()[..., None])  # (K, N, H)

		y_max_stt = torch.div(y_max - bboxes[:, 1:2], patch_pixel_size[..., None])
		max_coor = torch.le(coor, y_max_stt.long()[..., None])  # (K, N, H)

		visible_part_map = (min_coor * max_coor).unsqueeze(-1).repeat(1, 1, 1, bbox_features.shape[-1]).long()
		visible_part_map = visible_part_map * visible_part_indicator.unsqueeze(-1).unsqueeze(-1)  # (K, N, H, W)
		
		global_features = F.adaptive_avg_pool2d(bbox_features, output_size=1).squeeze((-1, -2))

		masked_features = bbox_features.unsqueeze(1) * visible_part_map.unsqueeze(2)
		part_features = masked_features.sum(dim=(-2, -1)) / visible_part_map.sum(dim=(-2, -1))[..., None].clamp(min=1e-6)

		global_feature_norm = F.normalize(global_features, p=2, dim=-1)
		part_feature_norm = F.normalize(part_features, p=2, dim=-1)

		all_features = torch.concat([global_feature_norm.unsqueeze(1), part_feature_norm], dim=1)
		# all_features = torch.concat([global_features.unsqueeze(1), part_features], dim=1)

		return all_features, visible_part_indicator

	def _update_classifier(self,
		all_features, kpts,
		visible_part_indicator,
		target_id,
	):
		K = kpts.shape[0]
		# update short-term memory		
		for k in range(K):
			if k == target_id:  # positive
				for n in range(self.N+1):
					if n == 0:
						if visible_part_indicator[k, n-1] == 0:
							continue
					
					if len(self.st_pos_memory[n]) > self.memory_size:
						self.st_pos_memory[n].pop(0)
					self.st_pos_memory[n].append(all_features[k, n])
			else:  # negative
				for n in range(self.N+1):
					if n == 0:
						if visible_part_indicator[k, n-1] == 0:
							continue
					
					if len(self.st_neg_memory[n]) > self.memory_size:
						self.st_neg_memory[n].pop(0)
					self.st_neg_memory[n].append(all_features[k, n])
			
		for k in range(K):
			if k == target_id:  # positive
				for n in range(self.N+1):
					if n != 0:
						if visible_part_indicator[k, n-1] == 0:
							continue
					
					if len(self.lt_pos_memory[n]) < self.memory_size:
						self.lt_pos_memory[n].append(all_features[k, n])
					else:
						rand_idx = np.random.randint(0, self.memory_size)
						self.lt_pos_memory[n][rand_idx] = all_features[k, n]
			else:  # negative
				for n in range(self.N+1):
					if n != 0:
						if visible_part_indicator[k, n-1] == 0:
							continue
					
					if len(self.lt_pos_memory[n]) < self.memory_size:
						self.lt_pos_memory[n].append(all_features[k, n])
					else:
						rand_idx = np.random.randint(0, self.memory_size)
						self.lt_pos_memory[n][rand_idx] = all_features[k, n]
							
		for idx, classifier in enumerate(self.classifiers):
			if len(self.st_pos_memory[idx]) < self.min_pos_example:
				continue
			
			X = torch.stack(self.st_pos_memory[idx]).cpu().detach().numpy()
			y = torch.ones(X.shape[0])
			
			if len(self.st_neg_memory[idx]) > 0:
				X_neg = torch.stack(self.st_neg_memory[idx]).cpu().detach().numpy()
				y_neg = -1 * torch.ones(X_neg.shape[0])
				X = np.concatenate([X, X_neg], axis=0)
				y = np.concatenate([y, y_neg], axis=0)
			
			classifier.fit(X, y)

	def yolo_callback(self, msg: ResultArray):
		image = self.cv_bridge.imgmsg_to_cv2(msg.image, "bgr8")
		_image = cv2.resize(image, (self.img_size[1], self.img_size[0]))
		# print("resolution:", image.shape, _image.shape)

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

		if len(bboxes) == 0: return

		bboxes = torch.tensor(bboxes).cuda()
		kpts = torch.tensor(kpts).cuda()
		confs = torch.tensor(confs).cuda()
		positions = torch.tensor(positions).cuda()

		crop_imgs = [
			_image[
				int(bbox[1]-bbox[3]//2):int(bbox[1]+bbox[3]//2), int(bbox[0]-bbox[2]//2):int(bbox[0]+bbox[2]//2)
			] for bbox in bboxes
		]
		all_features, visible_part_indicator = self._process_crops(
			bboxes, kpts, confs, crop_imgs
		)
		# print("all_features", all_features.shape)
		# print("visible_part_indicator", visible_part_indicator.shape)
		# (K, N+1, D), (K, N)

		if not self.init_flag: return
		# if not self.init_flag or self.cam_model is None: return

		if self.is_following:  # continue following
			scores = [
				classifier.predict(all_features[:, idx].cpu().detach().numpy())
				for idx, classifier in enumerate(self.classifiers)
			]
			avg_scores = np.mean(scores, axis=0)  # * visiblity_part_indicator
			target_id = np.argmax(avg_scores)
			
			# not good enough re-identification
			# if avg_scores[target_id] < 0.5:
			# 	self.get_logger().warn(f"Insufficient re-identification confidence {avg_scores[target_id]}")
			# 	return

			if positions[target_id] is None: return

			pose_stamp = self._point_to_posestamp(
				positions[target_id],
				msg.image.header
			)
			if pose_stamp is None: return
			self.update_publisher.publish(pose_stamp)
		else:  # start following
			H, W, _ = image.shape
			target_id = self._get_target_from_bboxes(bboxes, W, H)
			
			if positions[target_id] is None: return
			# initialize PoseStamped from user's position
			pose_stamp = self._point_to_posestamp(
				positions[target_id],
				msg.image.header
			)
			if pose_stamp is None: return

			# transform the PoseStamped to map's frame
			# map_pose_stamp = self._transform_pose(pose_stamp)
			# if map_pose_stamp is None: return

			goal_msg = NavigateToPose.Goal()
			goal_msg.pose = pose_stamp
			if not self.nav_client.wait_for_server(timeout_sec=1.0):
				self.get_logger().info(f"Navigation server is not online!")
				return FollowMe.Result()
			nav_future = self.nav_client.send_goal_async(goal_msg)
			self.is_following = True
			self.last_user_position = positions[target_id]

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
		# marker.scale.x = 0.2  # diameter in x
		# marker.scale.y = 1.0  # diameter in y
		# marker.scale.z = 0.2  # height
		marker.scale.x = 0.2 # diameter in x
		marker.scale.y = 0.2  # diameter in y
		marker.scale.z = 1.0  # height
		marker.lifetime.sec = 0  # 0 means forever

		marker.color = ColorRGBA(r=1.0, g=0.8, b=0.1, a=0.5)
		markers.markers.append(marker)
		self.marker_publisher.publish(markers)

		self._update_classifier(all_features, kpts, visible_part_indicator, target_id)

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