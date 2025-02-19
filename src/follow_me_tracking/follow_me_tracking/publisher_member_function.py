import os
from ament_index_python.packages import get_package_share_directory
import rclpy
from rclpy.node import Node

import numpy as np

from upo_laser_people_msgs.msg import PersonDetection, PersonDetectionList
from geometry_msgs.msg import Point


class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(PersonDetectionList, 'detections', 10)
        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.data_path = os.path.join(
            get_package_share_directory('follow_me_tracking'),
            "data", "Detections.npy"
        )
        
        self.counter = 0

    def timer_callback(self):
        data = np.array(np.load(self.data_path))[0:10, 0:150, :]
        self.get_logger().info(f'{self.counter} {data.shape}')

        msg = PersonDetectionList()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        for i, det in enumerate(data[:, self.counter]):
            # if i == 0:
            #     print(det[0], det[1])
            _person = PersonDetection()
            _person.type = 0
            _person.score = 1.

            _point = Point()
            _point.x = float(det[0])
            _point.y = 0.
            _point.z = float(det[1])
            _person.position = _point

            msg.detections.append(_person)
        self.publisher_.publish(msg)
        self.counter += 1
        if self.counter > data.shape[1]-1:
            self.counter = 0


def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()