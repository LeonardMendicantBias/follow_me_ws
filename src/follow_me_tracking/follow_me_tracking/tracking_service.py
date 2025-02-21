import rclpy
from rclpy.node import Node
import tf2_ros
from tf2_ros import TransformException

import numpy as np

from visualization_msgs.msg import Marker, MarkerArray
from upo_laser_people_msgs.msg import PersonDetection, PersonDetectionList

from .kalmanTracker import Tracker

import tf2_geometry_msgs
from geometry_msgs.msg import PoseStamped, Pose, Point
from follow_me_msgs.srv import PersonId, PersonTracking


class TrackingService(Node):

	def __init__(self):
		super().__init__('minimal_service')
		
		# publish the updated 
		self.tf_buffer = tf2_ros.Buffer()
		self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

		# an tracking module to associate robot with user
		self.subscription = self.create_subscription(
			PersonDetectionList,
			'detected_people',
			self.detection_callback,
			10
		)
		self.tracker = Tracker(1, 10, 5)

		self.visualizing_publisher = self.create_publisher(
			MarkerArray, 'tracked_people', 10
		)

		self.identification_service = self.create_service(
			PersonId, 'person_id', self.person_id
		)

		self.tracking_service = self.create_service(
			PersonTracking, 'person_tracking', self.person_tracking
		)

		self._track_colors = [
			(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
			(127, 127, 255), (255, 0, 255), (255, 127, 255),
			(127, 0, 255), (127, 0, 127),(127, 10, 255), (0,255, 127)
		]

	'''
		Update human tracks and send the updated positions to Nav
	'''
	def detection_callback(self, msg: PersonDetectionList):
		# if len(msg.detections) == 0: return
		detections = np.array([
			[det.position.x, det.position.y]
			for det in msg.detections
		])
		self.tracker.update(detections)

		_msg = MarkerArray()
		for idx, track in enumerate(self.tracker.tracks):
			marker = Marker()
			marker.header.frame_id = "velodyne"
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

			_idx = min(idx, len(self._track_colors) - 1)
			marker.color.r = self._track_colors[_idx][0] / 255.
			marker.color.g = self._track_colors[_idx][1] / 255.
			marker.color.b = self._track_colors[_idx][2] / 255.
			marker.color.a = 0.5

			_msg.markers.append(marker)
		self.visualizing_publisher.publish(_msg)

	def person_id(self, request, response):
		robot_posestamp = request.pose

		# transform robot pose from "map" to "velodyne" frame
		try:
			t = self.tf_buffer.lookup_transform(
				"velodyne", "map",
				rclpy.time.Time(),
				rclpy.duration.Duration(seconds=5.0)
			)
		except tf2_ros.TransformException as ex:
			print("error:", ex)
			return response
		
		transformed_posestamp = tf2_geometry_msgs.do_transform_pose_stamped(
			robot_posestamp, t
		)

		min_dist = np.inf
		for track in self.tracker.tracks:
			if len(track.trace) == 0:
				response.track_id = -1
				continue
			
			dist = np.linalg.norm(np.array([
				transformed_posestamp.pose.position.x,
				transformed_posestamp.pose.position.y,
			]) - np.array([
				track.trace[-1][0, 0],
				track.trace[-1][0, 1],
			]))
			if dist < min_dist:
				response.track_id = track.trackId
				min_dist = dist

		print("track_id:", response.track_id)

		return response
	
	def person_tracking(self, request, response):
		track_id = request.track_id

		# transform robot pose from "map" to "velodyne" frame
		try:
			t = self.tf_buffer.lookup_transform(
				"map", "velodyne",
				rclpy.time.Time(),
				rclpy.duration.Duration(seconds=5.0)
			)
		except tf2_ros.TransformException as ex:
			# response.pose = None
			return response
		
		_track_position = self.tracker.get_track_position(track_id)
		if _track_position is None:
			return response
		
		track_position = PoseStamped()
		track_position.header.frame_id = "velodyne"
		track_position.header.stamp = self.get_clock().now().to_msg()
		track_position.pose.position.x = _track_position[0]
		track_position.pose.position.y = _track_position[1]
		track_position.pose.position.z = 0.

		transformed_posestamp = tf2_geometry_msgs.do_transform_pose_stamped(
			track_position, t
		)
		transformed_posestamp.pose.orientation = t.transform.rotation

		print("tracking", transformed_posestamp)
		response.pose = transformed_posestamp
		return response


def main():
	rclpy.init()

	minimal_service = TrackingService()

	rclpy.spin(minimal_service)

	rclpy.shutdown()

# def _get_track_from_frame(self, frame_pose: PoseStamped):
# 	try:
# 		print("first")
# 		self.get_logger().info(f"Getting velodyne's transformation.")
# 		t = self.tf_buffer.lookup_transform(
# 			"map", "velodyne",
# 			rclpy.time.Time(),
# 			# 3 sec seems to be the minimum
# 			# timeout for 5 sec for safe-measure
# 			rclpy.duration.Duration(seconds=10.0)
# 		)
# 	except TransformException as ex:
# 		self.get_logger().info(
# 			f'Could not transformation for velodyne: {ex}'
# 		)
# 		return None
	
# 	track_id = None
# 	track_pose = None
# 	min_dist = np.inf
# 	for i, track in enumerate(self.tracker.tracks):
# 		track_position = PoseStamped()
# 		track_position.header.frame_id = "velodyne"
# 		track_position.header.stamp = self.get_clock().now().to_msg()
# 		track_position.pose.position.x = track.trace[-1][0, 0]
# 		track_position.pose.position.y = track.trace[-1][0, 1]
# 		track_position.pose.position.z = 0.
# 		track_position.pose.orientation = t.transform.rotation

# 		transformed_pose = tf2_geometry_msgs.do_transform_pose_stamped(
# 			track_position, t
# 		)
		
# 		dist = np.linalg.norm(np.array([
# 			frame_pose.pose.position.x,
# 			frame_pose.pose.position.y,
# 			frame_pose.pose.position.z,
# 		]) - np.array([
# 			transformed_pose.pose.position.x,
# 			transformed_pose.pose.position.y,
# 			transformed_pose.pose.position.z,
# 		]))
# 		# print(i, transformed_pose.pose.position, dist)
# 		if dist < min_dist:
# 			track_id = track.trackId
# 			min_dist = dist
# 			track_pose = transformed_pose

# 	print(f"found matching for robot with id: {track_id} and dist at {dist}")
# 	return track_id, track_pose

if __name__ == '__main__':
	main()