import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import cv2
import numpy as np
from pathlib import Path
import threading
from time import sleep
import os
import torch

class Zed2CustomNode(Node):
    def __init__(self):
        super().__init__('zed2_custom_node')

        # Initialize publishers
        self.rgb_pub = self.create_publisher(Image, '/zed2/rgb/image_rect_color', 10)
        self.stop_pub = self.create_publisher(Bool, '/stopflag', 10)

        # Subscribe to the ZED image topic
        self.create_subscription(Image, '/zed/zed_node/stereo/image_rect_color', self.image_callback, 10)

        # Initialize CvBridge
        self.bridge = CvBridge()
        
        # Initialize threading primitives
        self.get_logger().info("Zed2CustomNode initialized successfully.")
        self.lock = threading.Lock()
        self.exit_signal = threading.Event()

        # Placeholder for the latest image to process
        self.latest_image = None

    def image_callback(self, msg):
        """Callback function to process incoming images from the ZED topic."""
        try:
            # Convert the ROS Image message to OpenCV format with original encoding
            cv_image = self.bridge.imgmsg_to_cv2(msg)

            # Convert BGRA to BGR if needed
            if cv_image.shape[2] == 4:  # If image has 4 channels (BGRA)
                original_image = cv2.cvtColor(cv_image, cv2.COLOR_BGRA2BGR)
            else:
                original_image = cv_image.copy()  # Already in BGR format

            # Define desired dimensions
            desired_width = 1280
            desired_height = 360

            # Check image dimensions
            if original_image.shape[1] != desired_width or original_image.shape[0] != desired_height:
                self.get_logger().warn(f"Unexpected image size: {original_image.shape[1]}x{original_image.shape[0]}. Resizing to {desired_width}x{desired_height}.")
                original_image = cv2.resize(original_image, (desired_width, desired_height))

            # Check for red blobs in the original image
            red_blob_detected = self.detect_red_blob(original_image)

            # Publish the original image
            try:
                rgb_msg = self.bridge.cv2_to_imgmsg(original_image, encoding="bgr8")
                self.rgb_pub.publish(rgb_msg)
                self.get_logger().debug("Published original RGB image.")
            except Exception as e:
                self.get_logger().error(f"Failed to publish original RGB image: {e}")

            # Publish stop flag based on red blob detection
            self.publish_images(original_image, red_blob_detected)

        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

    def publish_images(self, annotated_image, stop_flag):
        """Publish the annotated image and stop flag."""
        try:
            # Ensure the annotated image is in BGR format
            if annotated_image.shape[2] == 4:  # If BGRA
                annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGRA2BGR)

            # Publish stop flag
            stop_msg = Bool()
            stop_msg.data = stop_flag
            self.stop_pub.publish(stop_msg)
            self.get_logger().debug(f"Published stop flag: {stop_flag}")  # Changed to debug
        except Exception as e:
            self.get_logger().error(f"Failed to publish stop flag: {e}")

    def __del__(self):
        """Clean up resources."""
        self.exit_signal.set()
        self.get_logger().info("Zed2CustomNode is shutting down.")

    def detect_red_blob(self, image):
        """Detect red blobs in the image."""
        # Convert the image to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define the range for red color in HSV
        lower_red = np.array([0, 100, 100])
        upper_red = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv_image, lower_red, upper_red)

        lower_red = np.array([160, 100, 100])
        upper_red = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv_image, lower_red, upper_red)

        # Combine masks
        red_mask = mask1 | mask2

        # Check if any red pixels are detected
        return np.any(red_mask)

def main(args=None):
    rclpy.init(args=args)
    node = Zed2CustomNode()
    node.get_logger().info("Zed2CustomNode has started.")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Zed2CustomNode interrupted by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        node.get_logger().info("Zed2CustomNode has been shut down.")

if __name__ == '__main__':
    main()