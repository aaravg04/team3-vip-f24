import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import copy
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
import math
import numpy as np

PI = math.pi
MIN_ANGLE = math.pi / 2
MAX_ANGLE = -math.pi / 2
DISPARITY_THRESHOLD = 0.2
C_W = 0.3
C_L = 0.5

class DisparityExtender:
    CAR_WIDTH = 0.3
    # the min difference between adjacent LiDAR points for us to call them disparate
    DIFFERENCE_THRESHOLD = 0.05
    MAX_SPEED = 2.0
    LINEAR_DISTANCE_THRESHOLD = 5.0
    ANGLE_CHANGE_THRESHOLD = 0.0
    ANGLE_CHANGE_SPEED = 0.5
    MAX_ANGLE = 0.8
    SLOW_SPEED = 1.0
    MAX_DISTANCE_C = 0.95
    WHEELBASE_WIDTH = 0.328  # 0.328
    coefficient_of_friction = 0.62
    gravity = 9.81998
    REVERSAL_THRESHHOLD = 0.85
    SLOWDOWN_SLOPE = 0.9

    prev_angle = 0.0
    prev_index = None
    is_reversing = False

    def __init__(self, logger):
        self.logger = logger

    def preprocess_lidar(self, ranges):
        """ Preprocess LiDAR data by removing certain quadrants. """
        eighth = int(len(ranges) / 6)
        return np.array(ranges[eighth:-eighth])

    def get_differences(self, ranges):
        """ Compute absolute differences between adjacent LiDAR points. """
        differences = [0.]  # set first element to 0
        for i in range(1, len(ranges)):
            differences.append(abs(ranges[i] - ranges[i - 1]))
        return differences

    def get_disparities(self, differences, threshold):
        """ Identify indices with significant differences. """
        disparities = []
        for index, difference in enumerate(differences):
            if difference > threshold:
                disparities.append(index)
        return disparities

    def get_num_points_to_cover(self, dist, width):
        """ Calculate the number of LiDAR points to cover based on distance and width. """
        angle_step = (0.25) * (math.pi / 180)  # 0.25 degrees in radians
        arc_length = angle_step * dist
        return int(math.ceil(self.CAR_WIDTH / arc_length))

    def cover_points(self, num_points, start_idx, cover_right, ranges):
        """ Cover a number of LiDAR points with a closer distance to prevent collisions. """
        new_dist = ranges[start_idx]
        if cover_right:
            for i in range(num_points):
                next_idx = start_idx + 1 + i
                if next_idx >= len(ranges):
                    break
                if ranges[next_idx] > new_dist:
                    ranges[next_idx] = new_dist
        else:
            for i in range(num_points):
                next_idx = start_idx - 1 - i
                if next_idx < 0:
                    break
                if ranges[next_idx] > new_dist:
                    ranges[next_idx] = new_dist
        return ranges

    def extend_disparities(self, disparities, ranges, car_width):
        """ Extend disparities to cover the car width. """
        width_to_cover = (car_width / 2)
        for index in disparities:
            first_idx = index - 1
            points = ranges[first_idx:first_idx + 2]
            close_idx = first_idx + np.argmin(points)
            far_idx = first_idx + np.argmax(points)
            close_dist = ranges[close_idx]
            num_points_to_cover = self.get_num_points_to_cover(close_dist, width_to_cover)
            cover_right = close_idx < far_idx
            ranges = self.cover_points(num_points_to_cover, close_idx, cover_right, ranges)
        return ranges

    def get_steering_angle(self, range_index, angle_increment, range_len):
        """ Calculate the steering angle corresponding to a LiDAR point. """
        angle = -1.57 + (range_index * angle_increment)

        if angle < -1.57:
            angle = -1.57
        elif angle > 1.57:
            angle = 1.57
        return angle

    def calculate_min_turning_radius(self, angle, forward_distance):
        """ Calculate the minimum turning radius based on steering angle and distance. """
        angle = abs(angle)
        if angle < 0.0872665:  # 5 degrees in radians
            return self.MAX_SPEED
        else:
            turning_radius = (self.WHEELBASE_WIDTH / math.sin(angle))
            maximum_velocity = math.sqrt(self.coefficient_of_friction * self.gravity * turning_radius)

            # Calculate stopping distance
            stopping_distance = (maximum_velocity ** 2) / (2 * 0.5 * self.gravity)
            if stopping_distance > forward_distance:
                # Reduce velocity to ensure stopping within forward distance
                maximum_velocity = math.sqrt(2 * 0.5 * self.gravity * forward_distance)

            if maximum_velocity < self.MAX_SPEED:
                maximum_velocity = maximum_velocity * (maximum_velocity / self.MAX_SPEED)
            else:
                maximum_velocity = self.MAX_SPEED

        return maximum_velocity

    def _process_lidar(self, lidar_data):
        """ Process LiDAR data to compute speed and steering angle. """
        ranges = lidar_data.ranges
        self.radians_per_point = (2 * np.pi) / len(ranges)
        proc_ranges = self.preprocess_lidar(ranges)
        differences = self.get_differences(proc_ranges)
        disparities = self.get_disparities(differences, self.DIFFERENCE_THRESHOLD)
        proc_ranges = self.extend_disparities(disparities, proc_ranges, self.CAR_WIDTH)

        # Find the maximum range and its index
        max_value = max(proc_ranges)
        max_index = np.argmax(proc_ranges)

        # Apply additional logic to find a better max_index if needed
        np_ranges = np.array(proc_ranges)
        greater_indices = np.where(np_ranges >= max_value * self.MAX_DISTANCE_C)[0]

        if greater_indices.size > 0:
            # Choose the index closest to the center
            center_index = len(proc_ranges) // 2
            differences = np.abs(greater_indices - center_index)
            max_index = greater_indices[np.argmin(differences)]
            max_value = proc_ranges[max_index]
        else:
            # Fallback to the original max_index if no greater_indices found
            pass

        # Calculate steering angle
        steering_angle = self.get_steering_angle(max_index, lidar_data.angle_increment, len(proc_ranges))

        d_theta = abs(steering_angle)

        self.prev_angle = steering_angle
        self.prev_index = max_index

        # Determine speed based on distance and other conditions
        if (self.is_reversing and max_value < 2.7) or (not self.is_reversing and max_value < 2):
            speed = -0.75
            steering_angle = -steering_angle
            self.is_reversing = True
        else:
            self.is_reversing = False
            speed_d = max(0.5, self.MAX_SPEED - self.MAX_SPEED * (self.SLOWDOWN_SLOPE * (self.LINEAR_DISTANCE_THRESHOLD - max_value) / self.LINEAR_DISTANCE_THRESHOLD))
            speed_a = self.calculate_min_turning_radius(steering_angle, max_value)
            speed = min(speed_d, speed_a)
            min_speed = 0.5

            if max_value < 0.5:
                min_speed = 0.3
            elif max_value < 1.3:
                min_speed = 0.5
            elif max_value < 1.7:
                min_speed = 0.75
            elif max_value < 2.0:
                min_speed = 1.0
            elif max_value < 2.5:
                min_speed = 1.3
            elif max_value < 3:
                min_speed = 1.7
            else:
                min_speed = 2.0

            speed = max(0.5, min_speed, speed)

        if speed > self.MAX_SPEED:
            speed = self.MAX_SPEED

        # ----- Speed Adjustment Based on Steering Angle -----
        # Introduce a scaling factor to reduce speed as the steering angle increases
        # Define maximum steering angle (in radians) for scaling (e.g., 90 degrees)
        MAX_STEERING_ANGLE_RAD = math.radians(90)  # 1.5708 radians

        # Calculate the scaling factor (e.g., linearly decrease speed)
        # Ensure that the scaling factor is between 0.5 and 1.0
        # When steering_angle is 0, scaling_factor = 1 (full speed)
        # When steering_angle is MAX_STEERING_ANGLE_RAD, scaling_factor = 0.5 (half speed)
        scaling_factor = 1.0 - (0.5 * (d_theta / MAX_STEERING_ANGLE_RAD))
        scaling_factor = max(0.5, min(1.0, scaling_factor))  # Clamp between 0.5 and 1.0

        # Apply the scaling factor to the speed
        speed *= scaling_factor

        # Optionally, ensure speed does not go below a minimum threshold
        MIN_SPEED = 0.3
        speed = max(speed, MIN_SPEED)

        # Log the scaling information for debugging
        self.logger.debug(f"Steering Angle: {steering_angle:.2f} rad, Scaling Factor: {scaling_factor:.2f}, Adjusted Speed: {speed:.2f}")

        # Ensure speed is within the allowed range after scaling
        speed = max(0.5, speed)
        if speed > self.MAX_SPEED:
            speed = self.MAX_SPEED

        self.logger.info(f"speed: {speed:.2f}, max_value: {max_value:.2f}, steering_angle: {steering_angle:.2f} rad")

        return speed, steering_angle, max_value, max_index, differences, disparities, proc_ranges

    def process_observation(self, lidar_data, ego_odom):
        return self._process_lidar(lidar_data)


class AckermannPublisher(Node):

    def __init__(self):
        super().__init__('team_1_publisher')
        self.disparity = DisparityExtender(self.get_logger())
        self.laser_subscription = self.create_subscription(
            LaserScan,  # message type
            'scan',
            self.lidar_callback,
            10)

        self.publisher = self.create_publisher(AckermannDriveStamped, '/ackermann_cmd', 10)

    def lidar_callback(self, msg: LaserScan):
        # Process LiDAR data to get speed and steering angle
        speed, angle, max_value, max_index, differences, disparities, proc_ranges = self.disparity._process_lidar(msg)
        stamped_msg = AckermannDriveStamped()
        stamped_msg.drive = AckermannDrive()
        stamped_msg.drive.steering_angle = angle
        stamped_msg.drive.speed = speed

        self.publisher.publish(stamped_msg)


def main(args=None):
    rclpy.init(args=args)
    ackermann_publisher_i = AckermannPublisher()
    rclpy.spin(ackermann_publisher_i)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    ackermann_publisher_i.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
