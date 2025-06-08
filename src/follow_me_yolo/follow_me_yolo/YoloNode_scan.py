import time
from pathlib import Path

import rclpy
from rclpy.node import Node

from message_filters import Subscriber, ApproximateTimeSynchronizer
from tf2_ros import Buffer, TransformListener, TransformException

from sensor_msgs.msg import Image, CameraInfo, LaserScan
from vision_msgs.msg import BoundingBox2D, Point2D
from geometry_msgs.msg import Pose2D, Point
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
      self.declare_parameter('img_width', 448)
      self.declare_parameter('img_height', 256)
      self.declare_parameter('conf_threshold', 0.5)

      self.img_size = [
         self.get_parameter('img_height').get_parameter_value().integer_value,
         self.get_parameter('img_width').get_parameter_value().integer_value
      ]

      self.cam_model = PinholeCameraModel()
      self.cv_bridge = CvBridge()

      self.info_sub = Subscriber(self, CameraInfo, '/camera/camera/color/camera_info')
      self.image_sub = Subscriber(self, Image, '/camera/camera/color/image_raw')
      self.scan_sub = Subscriber(self, LaserScan, '/scan')
      self.ts = ApproximateTimeSynchronizer(
         [self.info_sub, self.image_sub, self.scan_sub],
         queue_size=1, slop=0.1
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
      self.last_marker_count = 0

      self.is_ran = False

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

   def synced_callback(self,
      info_msg: CameraInfo,
      rgb_msg: Image,
      scan_msg: LaserScan,
   ):
      if self.is_ran:
         return
      
      print(scan_msg)
      self.is_ran = True
      image = self.cv_bridge.imgmsg_to_cv2(rgb_msg, "bgr8")  # (480, 848, 3)
      response = ResultArray()
      response.image = rgb_msg

      # perform AI functions
      results = self.model(
         image, self.img_size,
         conf=self.get_parameter('conf_threshold').get_parameter_value().double_value,
         verbose=False
      )
      result = list(results)[0]

      if len(result) == 0:
         self.result_publisher.publish(response)
         return

      xywh = result.boxes.xywh.cpu().numpy()
      conf = result.keypoints.conf.cpu().numpy()
      xy = result.keypoints.xy.cpu().numpy()

      # extract positions of detected humans from LaserScan
      # 1/ convert PCD from velodyne to camera frame
      # try:
      #    # Transform the ray into the scan frame
      #    point_scan = self.tf_buffer.transform(point_cam, self.scan_frame, rclpy.time.Time())
      # except Exception as e:
      #    self.get_logger().error(f"TF transform failed: {e}")
      #    return None

      #/ 2/ 


def main(args=None):
   rclpy.init(args=args)

   publisher = YoloPublisher()

   rclpy.spin(publisher)


if __name__ == '__main__':
   main()