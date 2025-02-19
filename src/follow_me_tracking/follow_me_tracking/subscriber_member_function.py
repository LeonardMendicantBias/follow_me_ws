import rclpy
from rclpy.node import Node
import numpy as np

from geometry_msgs.msg import Point
from upo_laser_people_msgs.msg import PersonDetection, PersonDetectionList
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import Header

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
        self.publisher_ = self.create_publisher(MarkerArray, 'track_people', 10)

    def listener_callback(self, msg: PersonDetectionList):
        # print(msg.header.frame_id)
        data = np.array([
            [det.position.x, det.position.z]
            for det in msg.detections
        ])
        self.tracker.update(data)


        # print(data.shape, len(self.tracker.tracks))

        msg = MarkerArray()
        for track_id in range(len(self.tracker.tracks)):
            marker = Marker()
            marker.header = Header()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()

            marker.id = track_id
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD

            marker.pose.position.y = self.tracker.tracks[track_id].trace[-1][0, 0] / 50
            marker.pose.position.x = self.tracker.tracks[track_id].trace[-1][0, 1] / 50

            marker.scale.x = 2.0*0.4
            marker.scale.y = 2.0*0.4
            marker.scale.z = 1.5

            marker.color.r = self.track_colors[self.tracker.tracks[track_id].trackId][0] / 255.
            marker.color.g = self.track_colors[self.tracker.tracks[track_id].trackId][1] / 255
            marker.color.b = self.track_colors[self.tracker.tracks[track_id].trackId][2] / 255.
            marker.color.a = 0.5

            msg.markers.append(marker)
        self.publisher_.publish(msg)


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