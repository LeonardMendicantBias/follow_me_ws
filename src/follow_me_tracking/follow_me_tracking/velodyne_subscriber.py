import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2

import numpy as np

import struct
import laspy
import open3d as o3d

from segment_lidar import samlidar, view


def pointcloud2_to_xyz(pointcloud2_msg):
    """Convert a ROS2 PointCloud2 message to a NumPy array."""
    point_step = pointcloud2_msg.point_step
    row_step = pointcloud2_msg.row_step
    data = pointcloud2_msg.data
    
    points = []
    for i in range(0, len(data), point_step):
        x, y, z = struct.unpack_from('fff', data, i)
        points.append((x, y, z))
    
    return np.array(points, dtype=np.float32)


class VelodyneSubscriber(Node):

    def __init__(self):
        
        super().__init__('velodyne_subscriber')
        self.get_logger().info("listening")
        self.create_subscription(
            PointCloud2,
            'velodyne_points',
            self.listener_callback,
            10
        )
        theta = np.pi / 2  # 90 degrees

        self.rotation = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta),  0],
            [0, 0, 1]
        ])

        # self.viewpoint = view.TopView()
        # self.viewpoint = view.PinholeView()
        # self.model = samlidar.SamLidar(ckpt_path="sam_vit_h_4b8939.pth")

        self.count = 0

    def listener_callback(self, msg: PointCloud2):
        if self.count == 10: return
        self.count += 1

        points = pc2.read_points(
            msg, 
            field_names=["x", "y", "z"],
            skip_nans=True,
            # reshape_organized_cloud=True
        )
        # print(points["x"].shape, points["y"].shape, points["z"].shape)
        points = np.stack([
            points["x"], points["y"], points["z"]
        ], axis=-1)
        points = points @ self.rotation.T
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud(f'./data/{self.count}.pcd', pcd)

        header = laspy.header.LasHeader(point_format=3, version='1.2')
        las = laspy.LasData(header)
        las.x, las.y, las.z = points[:, 0], points[:, 1], points[:, 2]
        las.write(f'./data/{self.count}.las')


def main(args=None):
    rclpy.init(args=args)

    velodyne_subscriber = VelodyneSubscriber()

    rclpy.spin(velodyne_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    velodyne_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()