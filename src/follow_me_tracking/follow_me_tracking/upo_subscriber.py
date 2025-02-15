import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Point
from upo_laser_people_msgs.msg import PersonDetection, PersonDetectionList


class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            PersonDetectionList,
            'detected_people',
            self.listener_callback,
            10
        )
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg: PersonDetectionList):
        print(msg.header.frame_id)
        for detection in msg.detections:
            position: Point = detection.position
            print(position.x, "-", position.y)
            pass
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