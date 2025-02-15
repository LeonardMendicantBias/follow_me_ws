import rclpy
from rclpy.node import Node
import numpy as np

from geometry_msgs.msg import Point
from upo_laser_people_msgs.msg import PersonDetection, PersonDetectionList

from .tracker import Tracker


class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            PersonDetectionList,
            # 'detected_people',
            'detections',
            self.listener_callback,
            10
        )
        self.subscription  # prevent unused variable warning
        self.tracker = Tracker(150, 30, 5)
        self.track_colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (127, 127, 255), (255, 0, 255), (255, 127, 255),
            (127, 0, 255), (127, 0, 127),(127, 10, 255), (0,255, 127)
        ]

    def listener_callback(self, msg: PersonDetectionList):
        # print(msg.header.frame_id)
        data = np.array([
            [det.position.x, det.position.z]
            for det in msg.detections
        ])
        self.tracker.update(data)
        for detection in msg.detections:
            position: Point = detection.position
            print(position.x, "-", position.z)
        print("-"*30)


def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = MinimalSubscriber()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()