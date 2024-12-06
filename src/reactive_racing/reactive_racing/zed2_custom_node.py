import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import cv2
import numpy as np

class ImageProcessorNode(Node):
    initial_published = False
    count = 0

    def __init__(self):
        super().__init__('image_processor_node')

        # Initialize logger attribute
        self.logger = self.get_logger()

        # Publisher to publish stop flag
        self.stop_pub = self.create_publisher(Bool, '/stopflag', 10)
        self.stop_ack = self.create_publisher(Bool, '/stop_ack', 10)
        self.redimg_pub = self.create_publisher(Image, '/redimg', 10)

        # Subscriber to the image topic
        self.image_subscriber = self.create_subscription(
            Image,
            '/zed/zed_node/stereo_raw/image_raw_color',  # Replace with your actual image topic
            self.image_callback,
            10
        )

        # Initialize CvBridge
        self.bridge = CvBridge()

        # Parameters for red blob detection
        self.lower_red1 = np.array([0, 120, 70])    # Increased saturation and value thresholds
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([160, 120, 70])
        self.upper_red2 = np.array([179, 255, 255])

        # Minimum number of red pixels to qualify as a blob
        self.min_blob_size = 3000   # Adjust based on your requirements

        self.logger.info('ImageProcessorNode has been initialized.')
        self.initial_published = False

    def image_callback(self, msg):
        # self.logger.info("Processing image")
        # self.count += 1
        # if self.count % 3 != 0:
        #     return

        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Get image dimensions
            height, width, _ = cv_image.shape
            self.logger.debug(f'Image dimensions: width={width}, height={height}')

            # Calculate the start and end rows for the middle third
            start_row = 0 * height // 3
            end_row = 1 * height // 2
            self.logger.debug(f'Middle third rows: start={start_row}, end={end_row}')

            # crop 10% each side horizontally
            percent = 0.1
            start_col = round(percent * width)
            end_col = round((1-percent) * width)

            # Crop the image to the middle third vertically
            middle_third = cv_image[start_row:end_row, :, :]
            # 360,1080,3
            # h,w,c
            # cv_image = [:180,:,:]

            # Convert the cropped image from BGR to HSV
            hsv_image = cv2.cvtColor(middle_third, cv2.COLOR_BGR2HSV)

            # Create masks for red color (handling the wrap-around in HSV hue for red)
            mask1 = cv2.inRange(hsv_image, self.lower_red1, self.upper_red1)
            mask2 = cv2.inRange(hsv_image, self.lower_red2, self.upper_red2)
            red_mask = cv2.bitwise_or(mask1, mask2)

            # Optional: Apply morphological operations to reduce noise
            kernel = np.ones((3, 3), np.uint8)
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=2)
            red_mask = cv2.dilate(red_mask, kernel, iterations=1)

            # Find contours in the mask
            contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Initialize flag
            red_blob_detected = False

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area >= self.min_blob_size:
                    red_blob_detected = True
                    # Shift contour coordinates back to the original image
                    # cnt_shifted = cnt.copy()
                    # cnt_shifted[:, :, 1] += start_row  # Adjust the y-coordinate
                    # Draw the contour on the original image for visualization
                    # cv2.drawContours(cv_image, [cnt_shifted], -1, (0, 255, 0), 2)
                    # Log the detection
                    # self.logger.info(f'Red blob detected with area: {area}')
                    # If you want to detect multiple blobs, don't break here
                    # break  # Uncomment to stop after first detection

            # Publish the stop flag
            stop_msg = Bool()
            stop_msg.data = red_blob_detected
            self.stop_pub.publish(stop_msg)
            # self.logger.info(f'Published stop flag: {red_blob_detected}')

            # Publish the annotated image to a topic called /redimg
            # img_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
            # self.redimg_pub.publish(img_msg)

            # Publish the initial processed flag once after the first image is processed
            if not self.initial_published:
                initial_msg = Bool()
                initial_msg.data = True
                self.stop_ack.publish(initial_msg)
                self.initial_published = True
                self.logger.info('Published initial processed flag.')

        except Exception as e:
            self.logger.error(f'Error processing image: {e}')

    def __del__(self):
        self.logger.info('ImageProcessorNode is shutting down.')

def main(args=None):
    rclpy.init(args=args)
    node = ImageProcessorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Destroy the node explicitly
        # Disconnect the ZED camera via the SDK
        self.zed.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
