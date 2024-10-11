import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool  # Import the Bool message type
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
from yolov5 import YOLOv5  # Import YOLOv5 directly

class Zed2CustomNode(Node):
    def __init__(self):
        super().__init__('zed2_custom_node')

        # Initialize publishers
        self.rgb_pub = self.create_publisher(Image, '/zed2/rgb/image_rect_color', 10)
        self.depth_pub = self.create_publisher(Image, '/zed2/depth/depth_registered', 10)
        self.yolo_pub = self.create_publisher(Image, '/yolo_image', 10)
        self.stop_pub = self.create_publisher(Bool, '/stopflag', 10)  # Publisher for stop flag

        # Subscribe to the ZED image topic
        self.create_subscription(Image, '/zed/zed_node/stereo/image_rect_color', self.image_callback, 10)

        # Publish False initially
        stop_msg = Bool()
        stop_msg.data = False
        self.stop_pub.publish(stop_msg)

        self.bridge = CvBridge()

        # Initialize YOLOv5 model
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Check for CUDA
        self.device = 'cuda'
        self.model = self.load_yolov5_model()  # Load YOLOv5 model
        print("INITIALIZED ! ! ! ! ! !!")

    def load_yolov5_model(self):
        """Load the YOLOv5 model."""
        model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)  # Load YOLOv5s model from Torch Hub
        model.to(self.device)  # Move model to the appropriate device (CPU or GPU)
        print(f"model on {self.device}")
        return model

    def image_callback(self, msg):
        """Callback function to process incoming images from the ZED topic."""
        # Convert the ROS Image message to a format suitable for YOLOv5
        rgb_image_data = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Ensure the input image is 1280x360
        if rgb_image_data.shape[1] == 1280 and rgb_image_data.shape[0] == 360:
            # Split the image into two 640x360 images
            left_image = rgb_image_data[:, :640, :]  # Left half
            right_image = rgb_image_data[:, 640:, :]  # Right half

            # Zero-pad each image to 640x640
            left_image_padded = cv2.copyMakeBorder(left_image, 0, 280, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            right_image_padded = cv2.copyMakeBorder(right_image, 0, 280, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

            # Run YOLOv5 object detection on each padded image
            detections_left = self.run_yolov5_object_detection(left_image_padded)
            detections_right = self.run_yolov5_object_detection(right_image_padded)

            # Combine detections from both images
            detections = detections_left + detections_right

            # Draw bounding boxes on the original image (or a combined image if needed)
            self.draw_detections(rgb_image_data, detections)

            # Publish images
            self.publish_images(rgb_image_data)
        else:
            print("Input image is not 1280x360, skipping processing.")

    def run_yolov5_object_detection(self, rgb_image_data):
        """Run object detection using YOLOv5."""
        # Ensure rgb_image_data is a numpy array with the correct type
        rgb_image_data = np.transpose(rgb_image_data, (2, 0, 1))  # Change shape from (H, W, C) to (C, H, W)
        rgb_image_data = np.expand_dims(rgb_image_data, axis=0)  # Add batch dimension; becomes (1, C, H, W)
        rgb_image_data = torch.from_numpy(rgb_image_data.astype('float')).to(self.device)  # Convert to tensor and move to device

        results = self.model(rgb_image_data)  # Run inference

        # Access detections based on the output format
        detections = results[0]  # Get detections from the results

        # Process detections
        detected_objects = []
        for detection in detections:
            # Extract bounding box information
            x_center, y_center, width, height, conf, *class_confidences = detection.tolist()

            # Check if the confidence is above a threshold (e.g., 0.5)
            if conf > 0.07:
                # Get the class index with the highest confidence
                class_index = class_confidences.index(max(class_confidences))
                class_label = class_index + 1  # Adjust for class indexing (if classes start from 1)

                if class_label == 11:  # Assuming class 11 is the stop sign
                    detected_objects.append({
                        'bounding_box': [x_center, y_center, width, height],
                        'confidence': conf,
                        'class_label': class_label
                    })

        return detected_objects  # Return the detected objects

    def draw_detections(self, rgb_image_data, detections):
        """Draw bounding boxes and labels on the image."""
        stop_sign_detected = False  # Flag to check if a stop sign is detected
        for detection in detections:
            if detection['confidence'] > 0.5:  # Confidence threshold
                stop_sign_detected = True  # Set flag to True if a stop sign is detected
                # Draw bounding box
                cv2.rectangle(rgb_image_data, (int(detection['bounding_box'][0]), int(detection['bounding_box'][1])),
                                          (int(detection['bounding_box'][2]), int(detection['bounding_box'][3])), (0, 255, 0), 2)

        # Publish stop flag
        stop_msg = Bool()
        stop_msg.data = stop_sign_detected
        self.stop_pub.publish(stop_msg)
        print("Published stop flag:", stop_sign_detected)  # Print message when stop flag is published

    def publish_images(self, rgb_image_data):
        """Publish the RGB images."""
        # Publish the YOLO image with bounding boxes
        yolo_msg = self.bridge.cv2_to_imgmsg(rgb_image_data, encoding="bgr8")
        self.yolo_pub.publish(yolo_msg)
        print("Published YOLO image with bounding boxes.")  # Print message when YOLO image is published

        # Publish the original RGB image
        rgb_msg = self.bridge.cv2_to_imgmsg(rgb_image_data, encoding="bgr8")
        self.rgb_pub.publish(rgb_msg)

    def __del__(self):
        """Clean up resources."""
        # No ZED resources to clean up
        pass

def main(args=None):
    rclpy.init(args=args)
    node = Zed2CustomNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()