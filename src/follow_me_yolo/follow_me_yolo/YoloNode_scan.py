import time
from pathlib import Path

import rclpy
from rclpy.node import Node

from message_filters import Subscriber, ApproximateTimeSynchronizer
from tf2_ros import Buffer, TransformListener, TransformException

from sensor_msgs.msg import Image, CameraInfo, LaserScan
from vision_msgs.msg import BoundingBox2D, Point2D
from geometry_msgs.msg import Pose2D, Point, Vector3Stamped
import sensor_msgs_py.point_cloud2 as pc2
import tf_transformations
from image_geometry import PinholeCameraModel
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA

import cv2
from cv_bridge import CvBridge, CvBridgeError

import numpy as np
from ultralytics import YOLO
import open3d as o3d

from follow_me_msgs.msg import ResultArray, YoloResult


def transform_point(point_3d_np, transform):
   """
   Transform a 3D point using a TF TransformStamped.
   
   Args:
      point_3d_np: numpy array [x, y, z]
      transform: TransformStamped (from tf2_ros.Buffer)
   
   Returns:
      Transformed point as numpy array [x, y, z]
   """
   # Extract translation
   t = transform.transform.translation
   translation = np.array([t.x, t.y, t.z])

   # Extract rotation (quaternion)
   q = transform.transform.rotation
   quaternion = [q.x, q.y, q.z, q.w]

   # Rotate point
   rotation_matrix = tf_transformations.quaternion_matrix(quaternion)[:3, :3]
   rotated_point = rotation_matrix @ point_3d_np

   # Apply translation
   transformed_point = rotated_point + translation

   return transformed_point


class YoloPublisher(Node):

   def __init__(self) -> None:
      super().__init__("yolo_node")
      self.tf_buffer = Buffer()
      self.tf_listener = TransformListener(self.tf_buffer, self)

      # self.declare_parameter("half", True)
      self.declare_parameter("frame_id", "base_footprint")
      self.declare_parameter('voxel_size', 0.1)

      self.declare_parameter('cluster_tolerance', 0.5)
      self.declare_parameter('min_cluster_size', 50)
      self.declare_parameter('max_cluster_size', 12500)
      self.declare_parameter('min_distance', 0.5)

      self.img_size = [256, 448]

      self.cam_model = PinholeCameraModel()
      self.cv_bridge = CvBridge()

      self.info_sub = Subscriber(self, CameraInfo, '/camera/camera/color/camera_info')
      self.image_sub = Subscriber(self, Image, '/camera/camera/color/image_raw')
      self.scan_sub = Subscriber(self, LaserScan, '/scan')
      self.ts = ApproximateTimeSynchronizer(
         [self.info_sub, self.image_sub, self.scan_sub],
         queue_size=1,
         slop=0.1  # maximum timestamp difference in seconds
      )
      self.ts.registerCallback(self.synced_callback)

      self.visualize_publisher = self.create_publisher(
         Image,
         '/yolo/visualization', 1
      )
      self.marker_publisher = self.create_publisher(
         MarkerArray,
         '/yolo/marker', 1
      )
      self.result_publisher = self.create_publisher(
         ResultArray,
         '/yolo', 1
      )

      self.conf_thres = 0.5
      _yolo_model = "yolo11n-pose"
      self.model = YOLO(_yolo_model)
      self.model.cuda()
      # if not Path(f"./{_yolo_model}.engine").is_file():
      #    self.model.export(
      #       format="engine",
      #       imgsz=self.img_size,
      #       half=True,
      #       simplify=True,
      #    )
      # self.tensorrt_model = YOLO(f"./{_yolo_model}.engine")
      self.get_logger().info("AI initiated.")

      self.skeleton = [
         [16, 14],
         [14, 12],
         [17, 15],
         [15, 13],
         [12, 13],
         [6, 12],
         [7, 13],
         [6, 7],
         [6, 8],
         [7, 9],
         [8, 10],
         [9, 11],
         [2, 3],
         [1, 2],
         [1, 3],
         [2, 4],
         [3, 5],
         [4, 6],
         [5, 7],
      ]
      pose_palette = np.array([
         [255, 128, 0],
         [255, 153, 51],
         [255, 178, 102],
         [230, 230, 0],
         [255, 153, 255],
         [153, 204, 255],
         [255, 102, 255],
         [255, 51, 255],
         [102, 178, 255],
         [51, 153, 255],
         [255, 153, 153],
         [255, 102, 102],
         [255, 51, 51],
         [153, 255, 153],
         [102, 255, 102],
         [51, 255, 51],
         [0, 255, 0],
         [0, 0, 255],
         [255, 0, 0],
         [255, 255, 255],
      ], dtype=np.uint8)
      self.limb_color = pose_palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
      self.kpt_color = pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
      self.count = 0
      self.last_marker_count = 0

   def visualize_yolo(self, image, result):
      if len(result.boxes) == 0:
         imgMsg = self.cv_bridge.cv2_to_imgmsg(image, 'bgr8')
         self.visualize_publisher.publish(imgMsg)
         return
      
      for bbox, kpts, confs in zip(result.boxes.xywh, result.keypoints.xy, result.keypoints.conf):
         x, y, w, h = map(int, bbox)
         cv2.rectangle(image, (x-w//2, y-h//2), (x+w//2, y+h//2), (0, 255, 0), 2)
         
         for j, (kpt, conf) in enumerate(zip(kpts, confs)):
            if conf < self.conf_thres:
               continue

            x, y = int(kpt[0]), int(kpt[1])
            cv2.circle(image, (x, y), 5, self.kpt_color[j].tolist(), -1)
         
         for i, sk in enumerate(self.skeleton):
            pos1 = (int(kpts[(sk[0] - 1), 0]), int(kpts[(sk[0] - 1), 1]))
            pos2 = (int(kpts[(sk[1] - 1), 0]), int(kpts[(sk[1] - 1), 1]))

            conf1 = confs[(sk[0] - 1)]
            conf2 = confs[(sk[1] - 1)]
            if conf1 < self.conf_thres or conf2 < self.conf_thres:
               continue
            # if pos1[0] % shape[1] == 0 or pos1[1] % shape[0] == 0 or pos1[0] < 0 or pos1[1] < 0:
            #     continue
            # if pos2[0] % shape[1] == 0 or pos2[1] % shape[0] == 0 or pos2[0] < 0 or pos2[1] < 0:
            #     continue
            cv2.line(
               image,
               pos1, pos2,
               self.limb_color[i].tolist(),
               thickness=3,
               lineType=cv2.LINE_AA,
            )

      imgMsg = self.cv_bridge.cv2_to_imgmsg(image, 'bgr8')
      self.visualize_publisher.publish(imgMsg)

   def visualize_marker(self):
      pass

   def laser_scan_to_points(self, scan_msg: LaserScan):
      angles = scan_msg.angle_min + np.arange(len(scan_msg.ranges)) * scan_msg.angle_increment
      ranges = np.array(scan_msg.ranges)
      
      # Remove NaNs and infs
      valid = np.isfinite(ranges)
      ranges = ranges[valid]
      angles = angles[valid]
      
      # Convert to 2D Cartesian coordinates
      xs = ranges * np.cos(angles)
      ys = ranges * np.sin(angles)
      
      points = np.stack([xs, ys, np.zeros_like(xs)], axis=1)  # (N, 3), z=0 in laser frame
      return points
   
   def find_closest_by_angle(self, points, ray_dir):
      norms = np.linalg.norm(points[:, :2], axis=1)
      unit_vectors = points[:, :2] / norms[:, None]

      dot_products = unit_vectors @ ray_dir
      best_idx = np.argmax(dot_products)  # Closest in angle (cosθ is max)
      return points[best_idx]
   
   def find_closest_by_perpendicular(self, points, ray_dir):
      projections = points[:, :2] @ ray_dir
      closest_points_on_ray = np.outer(projections, ray_dir)
      residuals = points[:, :2] - closest_points_on_ray
      distances = np.linalg.norm(residuals, axis=1)
      best_idx = np.argmin(distances)
      return points[best_idx]

   def direction_to_quaternion(self, vec, reference=np.array([1.0, 0.0, 0.0])):
      vec = vec / np.linalg.norm(vec)
      ref = reference / np.linalg.norm(reference)

      # Cross product gives rotation axis
      axis = np.cross(ref, vec)
      axis_norm = np.linalg.norm(axis)

      if axis_norm < 1e-8:
         # Vectors are parallel
         if np.dot(ref, vec) > 0:
               return np.array([0.0, 0.0, 0.0, 1.0])  # Identity quaternion
         else:
               # 180° rotation around orthogonal axis
               ortho = np.array([0.0, 0.0, 1.0]) if not np.allclose(ref, [0.0, 0.0, 1.0]) else np.array([0.0, 1.0, 0.0])
               axis = np.cross(ref, ortho)
               axis = axis / np.linalg.norm(axis)
               return tf_transformations.quaternion_about_axis(np.pi, axis)
      else:
         axis = axis / axis_norm
         angle = np.arccos(np.clip(np.dot(ref, vec), -1.0, 1.0))
         return tf_transformations.quaternion_about_axis(angle, axis)

   def synced_callback(self,
      info_msg: CameraInfo,
      rgb_msg: Image,
      scan_msg: LaserScan
   ):
      self.count += 1
      if self.count % 2 == 0:
         self.count = 0
         return
      # print(info_msg.header.frame_id, rgb_msg.header.frame_id, scan_msg.header.frame_id)
      # "camera_color_optical_frame", "camera_color_optical_frame", "velodyne"

      # process image
      image = self.cv_bridge.imgmsg_to_cv2(rgb_msg, "bgr8")  # (480, 848, 3)
      # _image = cv2.resize(image, (self.img_size[1], self.img_size[0]))
      # results = self.tensorrt_model(_image, self.img_size, verbose=False)
      # results = self.model(image, self.img_size, verbose=False)
      results = self.model(image, verbose=False)
      result = list(results)[0]
      self.visualize_yolo(image, result)
      response = ResultArray()
      response.image = rgb_msg
      response.image.header.frame_id = scan_msg.header.frame_id
      
      markers = MarkerArray()

      if len(result.boxes) == 0:
         self.result_publisher.publish(response)
         return
      self.get_logger().info(f"detecting {len(result.boxes)} humans")
      
      try:
         transform = self.tf_buffer.lookup_transform(
            scan_msg.header.frame_id, # target
            rgb_msg.header.frame_id,  # source
            rgb_msg.header.stamp,
            timeout=rclpy.duration.Duration(seconds=0.2)
         )
         q = transform.transform.rotation
         quaternion = [q.x, q.y, q.z, q.w]
         rotation_matrix = tf_transformations.quaternion_matrix(quaternion)  # 4x4 matrix
         rotation_only = rotation_matrix[:3, :3]
      except Exception as e:
         self.get_logger().warn(f"Skip due to missing transformation: {e}")
         return

      _time_stamp = self.get_clock().now().to_msg()
      self.cam_model.fromCameraInfo(info_msg)
      points = self.laser_scan_to_points(scan_msg)
      # print(points.shape)
      # points = points[points[:, 0] > 0]
      for idx_, (bbox, kpts, confs) in enumerate(zip(result.boxes.xywh, result.keypoints.xy, result.keypoints.conf)):
         kpts = kpts.cpu().tolist()
         x, y, w, h = bbox.cpu().tolist()
         ret = YoloResult()
         ret.bbox.center.position.x = x
         ret.bbox.center.position.y = y
         ret.bbox.center.theta = 0.
         ret.bbox.size_x = w
         ret.bbox.size_y = h
         ret.kpts = [
            Point2D(x=kpt[0], y=kpt[1])
            for kpt in kpts
         ]
         ret.confidences = [conf for conf in confs.cpu().tolist()]
         
         # extracting 3D positions
         ray = self.cam_model.projectPixelTo3dRay((x, y))
         transform_ray = rotation_only @ np.array(ray)
         ray_direction = transform_ray[:2] / np.linalg.norm(transform_ray[:2])
         point = self.find_closest_by_perpendicular(points, ray_direction)
         print("point", point)

         ret.position = Point(x=point[0], y=point[1], z=0.0)
         response.results.append(ret)

         marker = Marker()
         marker.header.frame_id = scan_msg.header.frame_id
         marker.header.stamp = _time_stamp
         
         marker.ns = 'cylinders'
         marker.id = idx_
         marker.type = Marker.CUBE
         marker.action = Marker.ADD
         marker.pose.position.x = point[0]
         marker.pose.position.y = point[1]
         marker.pose.position.z = point[2]
         marker.pose.orientation.w = 1.0  # No rotation
         marker.scale.x = 0.2  # diameter in x
         marker.scale.y = 0.2  # diameter in y
         marker.scale.z = 1.0  # height
         marker.lifetime.sec = 0  # 0 means forever

         marker.color = ColorRGBA(r=0.1, g=0.8, b=0.1, a=0.5)
         markers.markers.append(marker)
      
      for i in range(len(result.boxes.xywh), self.last_marker_count):
         marker = Marker()
         marker.header.frame_id = scan_msg.header.frame_id
         marker.header.stamp = _time_stamp
         marker.ns = 'cylinders'
         marker.id = i
         marker.action = Marker.DELETE
         markers.markers.append(marker)
      
      self.last_marker_count = len(result.boxes.xywh)

      self.marker_publisher.publish(markers)
      self.result_publisher.publish(response)
      # publish result


def main(args=None):
   rclpy.init(args=args)

   publisher = YoloPublisher()

   rclpy.spin(publisher)


if __name__ == '__main__':
   main()