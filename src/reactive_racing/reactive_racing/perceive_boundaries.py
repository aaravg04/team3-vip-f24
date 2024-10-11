
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from nav_msgs.msg import Odometry, Path
import sensor_msgs.msg
from sensor_msgs.msg import LaserScan, PointCloud2
from std_msgs.msg import Float64
from geometry_msgs.msg import PoseStamped
from ackermann_msgs.msg import AckermannDriveStamped
import time
import numpy as np



class BoundaryPerceptionNode(Node):

    def __init__(self):
        super().__init__('boundary_perception')

        qos_profile = QoSProfile(
            reliability = QoSReliabilityPolicy.RELIABLE,
            history = QoSHistoryPolicy.KEEP_LAST,
            durability = QoSDurabilityPolicy.TRANSIENT_LOCAL,
            depth=1
        )

        self.lidar_sub = self.create_subscription(
            LaserScan,
            "/scan",
            self.lidar_cb,
            1
        )

        self.point_cloud_publisher = self.create_publisher(
            PointCloud2,
            "/boundaries",
            qos_profile = qos_profile
        )

    def publish_point_cloud(self, xs, ys):
        pc_msg = PointCloud2()
        pc_msg.header.stamp = rclpy.time.Time().to_msg()
        pc_msg.header.frame_id = "laser"

        ros_dtype = sensor_msgs.msg.PointField.FLOAT32
        dtype = np.float32
        itemsize = np.dtype(dtype).itemsize

        zs = np.zeros_like(xs)
        points = np.vstack((xs, ys, zs)).T
        data = points.astype(dtype).tobytes()
        fields = [sensor_msgs.msg.PointField(name=n, offset=i*itemsize, datatype=ros_dtype, count=1) for i, n in enumerate('xyz')]

        pc_msg.height = 1
        pc_msg.width = points.shape[0]
        pc_msg.is_dense = False
        pc_msg.is_bigendian = False
        pc_msg.fields = fields
        pc_msg.point_step = itemsize * 3
        pc_msg.row_step = itemsize * 3 * points.shape[0]
        pc_msg.data = data
        self.point_cloud_publisher.publish(pc_msg)

    def lidar_cb(self, scan):
        # print(scan)
        # scan msg format: https://docs.ros.org/en/ros2_packages/rolling/api/sensor_msgs/interfaces/msg/LaserScan.html
        # TODO process scan
        xs = np.arange(10)
        ys = np.arange(10)
        self.publish_point_cloud(xs, ys)


def main(args=None):

    rclpy.init(args=args)
    perception = BoundaryPerceptionNode()
    rclpy.spin(perception)

    
    perception.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
