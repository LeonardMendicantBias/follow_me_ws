import time
from pathlib import Path

import rclpy
from rclpy.node import Node

from message_filters import Subscriber, ApproximateTimeSynchronizer
from tf2_ros import Buffer, TransformListener, TransformException

from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import BoundingBox2D, Pose2D, Point2D
from geometry_msgs.msg import Point, Pose, Vector3
import sensor_msgs_py.point_cloud2 as pc2
import tf_transformations
from image_geometry import PinholeCameraModel
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, Header

import cv2
from cv_bridge import CvBridge, CvBridgeError

import numpy as np
from ultralytics import YOLO
import open3d as o3d

from follow_me_msgs.msg import ResultArray, YoloResult


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

      self.rgb_model = PinholeCameraModel()
      # self.depth_model = PinholeCameraModel()
      self.cv_bridge = CvBridge()

      self.image_sub = Subscriber(self, Image, '/camera/camera/color/image_raw')
      self.depth_sub = Subscriber(self, Image, '/camera/camera/aligned_depth_to_color/image_raw')
      self.rgb_info_sub = Subscriber(self, CameraInfo, '/camera/camera/color/camera_info')
      # self.depth_info_sub = Subscriber(self, CameraInfo, '/camera/camera/depth/camera_info')
      self.ts = ApproximateTimeSynchronizer(
         [self.image_sub, self.depth_sub, self.rgb_info_sub], #, self.depth_info_sub],
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

   def visualize_yolo(self, image, result):
      if len(result.boxes) == 0:
         # imgMsg = self.cv_bridge.cv2_to_imgmsg(image, 'bgr8')
         imgMsg = self.cv_bridge.cv2_to_imgmsg(image, 'passthrough')
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

      # imgMsg = self.cv_bridge.cv2_to_imgmsg(image, 'bgr8')
      imgMsg = self.cv_bridge.cv2_to_imgmsg(image, 'passthrough')
      self.visualize_publisher.publish(imgMsg)

   def visualize_marker(self,
      frame_id,
      positions=[]
   ):
      _time_stamp = self.get_clock().now().to_msg()
      
      markers = MarkerArray()
      markers.markers = [
         Marker(
            header=Header(frame_id=frame_id, stamp=_time_stamp),
            ns="yolo", id=idx,
            type=Marker.CUBE, action=Marker.ADD,
            pose=Pose(position=Point(x=pos[0], y=pos[1], z=pos[2])),
            scale=Vector3(x=0.2, y=-1.0, z=0.2),
            color=ColorRGBA(r=0.1, g=0.8, b=0.1, a=0.5)
         )
         for idx, pos in enumerate(positions)
      ]
      if self.last_marker_count != len(positions):
         markers.markers.extend([
            Marker(
               header=Header(frame_id=frame_id, stamp=_time_stamp),
               ns="yolo", id=idx+len(positions),
               action=Marker.DELETE
            ) for idx in range(len(positions), self.last_marker_count)
         ])

      self.last_marker_count = len(positions)
      self.marker_publisher.publish(markers)

   def report(self,
      img_msg: Image,
      bboxes=[],
      kpts=[],
      confs=[],
      positions=[]
   ):
      response = ResultArray()
      response.image = img_msg
      response.results = [
         YoloResult(
            bbox=BoundingBox2D(
               center=Pose2D(position=Point2D(x=float(bbox[0]), y=float(bbox[1]))),
               size_x=float(bbox[2]), size_y=float(bbox[3])
            ),
            position=Point(x=pos[0], y=pos[1], z=pos[2]),
            # kpts=[Point2D(x=float(kpt[0]), y=float(kpt[1])) for kpt in kpts.tolist()],
            kpts=[Point2D(x=k[0], y=k[1]) for k in kpt.tolist()],
            confidences=conf.tolist(),
         )
         for bbox, kpt, conf, pos in zip(bboxes, kpts, confs, positions)
      ]

      self.result_publisher.publish(response)

   def synced_callback(self,
      rgb_msg: Image,
      depth_msg: Image,
      rgb_info_msg: CameraInfo,
      # depth_info_msg: CameraInfo,
   ):
      self.rgb_model.fromCameraInfo(rgb_info_msg)
      # self.depth_model.fromCameraInfo(depth_info_msg)

      image = self.cv_bridge.imgmsg_to_cv2(rgb_msg, "bgr8")  # (480, 848, 3)
      depth_image = self.cv_bridge.imgmsg_to_cv2(depth_msg, "passthrough")  # (480, 848)
      _depth_image = cv2.resize(depth_image, (image.shape[1], image.shape[0]))

      results = self.model(
         image, self.img_size,
         conf=self.get_parameter('conf_threshold').get_parameter_value().double_value,
         verbose=False
      )
      result = list(results)[0]
      self.get_logger().info(f"detecting {len(result.boxes)} humans")

      if len(result) == 0:
         self.report(rgb_msg)
         # self.visualize_yolo(_depth_image, result)
         self.visualize_yolo(image, result)
         return

      # (K, .)
      bboxes = result.boxes.xywh.cpu().numpy().astype(float)
      confs = result.keypoints.conf.cpu().numpy().astype(float)
      kpts = result.keypoints.xy.cpu().numpy().astype(float)  # (K, 17, 2)
      _xy = np.trunc(kpts).astype(int)
      _xy = _xy.clip([0, 0], [image.shape[1]-1, image.shape[0]-1])
      N = kpts.shape[1]
      
      distances = _depth_image[_xy[..., 1], _xy[..., 0]]
      is_kpts_vis = confs > self.conf_thres
      if not is_kpts_vis.any():
         self.report(rgb_msg)
         self.visualize_yolo(image, result)
         return
      
      distances = distances[is_kpts_vis.max(-1)]
      K = distances.shape[0]
      
      # print(is_kpts_vis)
      # print(is_kpts_vis.max(-1))
      # print(_xy.shape, _xy[is_kpts_vis.max(-1)].shape, _xy[is_kpts_vis.max(-1)])
      rays = np.array([
         self.rgb_model.projectPixelTo3dRay((u, v))
         for u, v in _xy[is_kpts_vis.max(-1)].reshape(K*N, -1)
      ]).reshape(K, N, -1)  # (K, N, 3)
      points_3d = rays * distances[..., None] / 1000.0

      try:
         transform = self.tf_buffer.lookup_transform(
            rgb_msg.header.frame_id,
            depth_msg.header.frame_id,
            rgb_msg.header.stamp,
            timeout=rclpy.duration.Duration(seconds=0.2)
         )
         t = transform.transform.translation
         translation = np.array([t.x, t.y, t.z])
         q = transform.transform.rotation
         quaternion = [q.x, q.y, q.z, q.w]
         rotation_matrix = tf_transformations.quaternion_matrix(quaternion)[:3, :3]
      except Exception as e:
         self.get_logger().warn(f"Skip due to missing transformation: {e}")
         return

      _is_kpts_vis = is_kpts_vis[is_kpts_vis.max(-1)]
      positions = (rotation_matrix @ points_3d.reshape(K*N, -1).T).T + translation
      positions = positions.reshape(K, N, -1) * _is_kpts_vis[..., None]
      positions = positions.sum(axis=-2) / _is_kpts_vis.sum(axis=-1, keepdims=True)
      
      self.report(rgb_msg,
         bboxes[is_kpts_vis.max(-1)], kpts[is_kpts_vis.max(-1)],
         confs[is_kpts_vis.max(-1)], positions#[is_kpts_vis.max(-1)]
      )
      # self.visualize_yolo(_depth_image, result)
      self.visualize_yolo(image, result)
      self.visualize_marker(rgb_msg.header.frame_id, positions)


def main(args=None):
   rclpy.init(args=args)

   publisher = YoloPublisher()

   rclpy.spin(publisher)


if __name__ == '__main__':
   main()