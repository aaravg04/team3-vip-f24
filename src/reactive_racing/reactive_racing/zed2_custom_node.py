#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import threading
from time import sleep
import os

class Zed2CustomNode(Node):
    def __init__(self):
        super().__init__('zed2_custom_node')

        # Initialize publishers
        self.rgb_pub = self.create_publisher(Image, '/zed2/rgb/image_rect_color', 10)
        self.yolo_pub = self.create_publisher(Image, '/yolo_image', 10)
        self.stop_pub = self.create_publisher(Bool, '/stopflag', 10)

        # Subscribe to the ZED image topic
        self.create_subscription(Image, '/zed/zed_node/stereo/image_rect_color', self.image_callback, 10)

        self.bridge = CvBridge()

        # Initialize YOLOv8 model
        self.model = self.load_yolov8_model(engine_file='yolov8n.engine')

        # Initialize threading primitives
        self.lock = threading.Lock()
        self.run_signal = threading.Event()
        self.exit_signal = threading.Event()
        self.detections = []

        # Placeholder for the latest image to process
        self.latest_image = None

        # Start the inference thread
        self.inference_thread = threading.Thread(target=self.inference_loop)
        self.inference_thread.start()

        self.get_logger().info("Zed2CustomNode initialized successfully.")

    def load_yolov8_model(self, engine_file='yolov8n.engine'):
        """Load and export the YOLOv8 TensorRT model."""
        try:
            # Check if the TensorRT engine file already exists
            if not os.path.exists(engine_file):
                self.get_logger().info(f"TensorRT engine file '{engine_file}' not found. Exporting now...")
                
                # Load the YOLOv8 PyTorch model
                model = YOLO("yolov8n.pt")
                self.get_logger().info("Loaded YOLOv8 PyTorch model.")

                # Export the model to TensorRT format
                model.export(format="engine", engine=engine_file, device=0)  # Creates 'yolov8n.engine'
                self.get_logger().info(f"Exported YOLOv8 model to '{engine_file}'.")

            else:
                self.get_logger().info(f"TensorRT engine file '{engine_file}' already exists.")

            # Load the exported TensorRT model
            trt_model = YOLO(engine_file)
            self.get_logger().info(f"TensorRT model loaded from '{engine_file}' on device: {trt_model.device}")
            return trt_model
        except Exception as e:
            self.get_logger().error(f"Failed to load and export YOLOv8 model: {e}")
            raise e

    def image_callback(self, msg):
        """Callback function to process incoming images from the ZED topic."""
        try:
            # Determine encoding from the ROS Image message
            encoding = msg.encoding
            
            # Convert the ROS Image message to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding=encoding)
            
            # Handle color space conversion based on encoding
            if 'bgra' in encoding.lower():
                # Convert BGRA to RGB
                rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGRA2RGB)
            elif 'bgr' in encoding.lower():
                # Convert BGR to RGB
                rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            else:
                self.get_logger().warn(f"Unhandled image encoding: {encoding}. Skipping conversion.")
                return
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        # Keep a copy of the original image for publishing
        original_image = cv_image.copy()

        # Define desired dimensions
        desired_width = 1280
        desired_height = 360

        # Check image dimensions
        if cv_image.shape[1] != desired_width or cv_image.shape[0] != desired_height:
            self.get_logger().warn(f"Unexpected image size: {cv_image.shape[1]}x{cv_image.shape[0]}. Resizing to {desired_width}x{desired_height}.")
            cv_image = cv2.resize(cv_image, (desired_width, desired_height))

        # Split the image into left and right halves
        left_image = cv_image[:, :640, :]  # Left half
        right_image = cv_image[:, 640:, :]  # Right half

        # Resize images to model input size (assuming 640x640 as per sample)
        input_size = (640, 640)
        left_image_resized = cv2.resize(left_image, input_size)
        right_image_resized = cv2.resize(right_image, input_size)

        # Acquire lock and update the latest image for inference
        with self.lock:
            self.latest_image = {
                'left': left_image_resized,
                'right': right_image_resized,
                'original': original_image
            }
            self.run_signal.set()  # Signal the inference thread to run

        # Publish the original image
        try:
            rgb_msg = self.bridge.cv2_to_imgmsg(original_image, encoding="bgr8")
            self.rgb_pub.publish(rgb_msg)
            self.get_logger().info("Published original RGB image.")
        except Exception as e:
            self.get_logger().error(f"Failed to publish original RGB image: {e}")

    def inference_loop(self):
        """Separate thread for running model inference."""
        self.get_logger().info("Inference thread started.")
        while not self.exit_signal.is_set():
            if self.run_signal.is_set():
                with self.lock:
                    if self.latest_image is not None:
                        left_image = self.latest_image['left']
                        right_image = self.latest_image['right']
                        original_image = self.latest_image['original']
                        self.latest_image = None  # Reset after copying

                # Run inference on both images within torch.no_grad()
                with torch.no_grad():
                    detections_left = self.run_yolov8_object_detection(left_image)
                    detections_right = self.run_yolov8_object_detection(right_image)

                # Combine detections from both images
                combined_detections = detections_left + detections_right

                # Draw detections on the original image
                annotated_image = original_image.copy()
                stop_sign_detected = self.draw_detections(annotated_image, combined_detections)

                # Publish the annotated image and stop flag
                self.publish_images(annotated_image, stop_sign_detected)

                # Reset the run signal
                self.run_signal.clear()

            sleep(0.01)  # Prevent busy waiting

        self.get_logger().info("Inference thread exiting.")

    def xywh2abcd(self, xywh, im_shape):
        """Convert bounding box from center-based (x, y, w, h) to corner coordinates (x1, y1, x2, y2)."""
        x_center, y_center, width, height = xywh
        x_min = (x_center - 0.5 * width) * im_shape[1]
        x_max = (x_center + 0.5 * width) * im_shape[1]
        y_min = (y_center - 0.5 * height) * im_shape[0]
        y_max = (y_center + 0.5 * height) * im_shape[0]

        # Ensure coordinates are within image bounds
        x_min = max(0, x_min)
        x_max = min(im_shape[1], x_max)
        y_min = max(0, y_min)
        y_max = min(im_shape[0], y_max)

        return [x_min, y_min, x_max, y_max]

    def run_yolov8_object_detection(self, image):
        """Run object detection using YOLOv8."""
        # Preprocess the image
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized_image = cv2.resize(rgb_image, (640, 640))
        normalized_image = resized_image.astype(np.float32) / 255.0
        input_tensor = np.transpose(normalized_image, (2, 0, 1))  # (C, H, W)
        input_tensor = np.expand_dims(input_tensor, axis=0)  # (1, C, H, W)
        input_tensor = torch.from_numpy(input_tensor).float().to('cuda')  # Ensure tensor is on GPU

        # Run inference using model.predict
        try:
            results = self.model.predict(
                input_tensor,
                save=False,
                imgsz=640,
                conf=0.4,  # Confidence threshold as per sample
                iou=0.45    # IoU threshold as per sample
            )
        except Exception as e:
            self.get_logger().error(f"Model inference failed: {e}")
            return []

        # Parse detections
        detected_objects = []
        for result in results:
            # Assuming result.boxes is available and contains the detections
            if hasattr(result, 'boxes') and result.boxes is not None:
                for box in result.boxes:
                    xywh = box.xywh.cpu().numpy().flatten()  # (x_center, y_center, width, height)
                    abcd = self.xywh2abcd(xywh, image.shape)
                    conf = box.conf.cpu().numpy().flatten()[0]
                    cls = int(box.cls.cpu().numpy().flatten()[0])

                    if conf > 0.5 and cls == 11:  # Adjust confidence threshold and class as needed
                        detected_objects.append({
                            'bounding_box': abcd,
                            'confidence': conf,
                            'class_label': cls
                        })

        return detected_objects

    def draw_detections(self, image, detections):
        """Draw bounding boxes and labels on the image."""
        stop_sign_detected = False  # Flag to check if a stop sign is detected
        for detection in detections:
            if detection['confidence'] > 0.5:  # Confidence threshold
                stop_sign_detected = True  # Set flag to True if a stop sign is detected

                x1, y1, x2, y2 = detection['bounding_box']
                # Draw bounding box
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                # Add label and confidence
                label = f"Stop Sign: {detection['confidence']:.2f}"
                cv2.putText(image, label, (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return stop_sign_detected

    def publish_images(self, annotated_image, stop_flag):
        """Publish the annotated image and stop flag."""
        try:
            # Publish YOLO annotated image
            yolo_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding="bgr8")
            self.yolo_pub.publish(yolo_msg)
            self.get_logger().info("Published YOLO image with bounding boxes.")

            # Publish stop flag
            stop_msg = Bool()
            stop_msg.data = stop_flag
            self.stop_pub.publish(stop_msg)
            self.get_logger().info(f"Published stop flag: {stop_flag}")
        except Exception as e:
            self.get_logger().error(f"Failed to publish images or stop flag: {e}")

    def __del__(self):
        """Clean up resources."""
        self.exit_signal.set()
        self.inference_thread.join()
        self.get_logger().info("Zed2CustomNode is shutting down.")

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
