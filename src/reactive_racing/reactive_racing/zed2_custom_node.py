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
        self.yolo_pub = self.create_publisher(Image, '/yolo_image', 10)
        self.stop_pub = self.create_publisher(Bool, '/stopflag', 10)

        # Subscribe to the ZED image topic
        self.create_subscription(Image, '/zed/zed_node/stereo/image_rect_color', self.image_callback, 10)

        # Initialize CvBridge
        self.bridge = CvBridge()
        
        # Load YOLOv8 using OpenCV DNN
        try:
            weights_path = "yolov8n.onnx"  # Make sure this file exists
            if not Path(weights_path).exists():
                self.get_logger().error(f"Model file {weights_path} not found!")
                raise FileNotFoundError(f"Model file {weights_path} not found!")
                
            self.model = cv2.dnn.readNetFromONNX(weights_path)
            
            # Use CUDA if available
            self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            
            self.get_logger().info("Model loaded successfully using OpenCV DNN")
            
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            raise e

        # Initialize threading primitives
        self.get_logger().info("Zed2CustomNode initialized successfully.")
        self.lock = threading.Lock()
        self.run_signal = threading.Event()
        self.exit_signal = threading.Event()
        self.detections = []

        # Placeholder for the latest image to process
        self.latest_image = None

        # Start the inference thread
        self.inference_thread = threading.Thread(target=self.inference_loop)
        self.inference_thread.start()

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

            # Split the image into left and right halves
            left_image = original_image[:, :640, :]  # Left half
            right_image = original_image[:, 640:, :]  # Right half

            # Preserve original padding logic: Zero-pad each image to 640x640
            left_image_padded = cv2.copyMakeBorder(
                left_image,
                top=0,
                bottom=280,  # 640 - 360 = 280
                left=0,
                right=0,
                borderType=cv2.BORDER_CONSTANT,
                value=(0, 0, 0)
            )
            right_image_padded = cv2.copyMakeBorder(
                right_image,
                top=0,
                bottom=280,  # 640 - 360 = 280
                left=0,
                right=0,
                borderType=cv2.BORDER_CONSTANT,
                value=(0, 0, 0)
            )

            # Acquire lock and update the latest image for inference
            with self.lock:
                self.latest_image = {
                    'left': left_image_padded,
                    'right': right_image_padded,
                    'original': original_image
                }
                self.run_signal.set()  # Signal the inference thread to run

            # Publish the original image
            try:
                rgb_msg = self.bridge.cv2_to_imgmsg(original_image, encoding="bgr8")
                self.rgb_pub.publish(rgb_msg)
                self.get_logger().debug("Published original RGB image.")
            except Exception as e:
                self.get_logger().error(f"Failed to publish original RGB image: {e}")

        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

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

    def xywh2abcd(self, xywh, shape):
        """Convert bbox from [x, y, w, h] to [a, b, c, d] format."""
        img_h, img_w = shape[:2]
        
        # Denormalize coordinates if they're normalized
        x, y, w, h = xywh
        if x <= 1 and y <= 1:  # If coordinates are normalized
            x *= img_w
            y *= img_h
            w *= img_w
            h *= img_h
        
        # Calculate corners
        x1 = max(0, int(x - w/2))
        y1 = max(0, int(y - h/2))
        x2 = min(img_w, int(x + w/2))
        y2 = min(img_h, int(y + h/2))
        
        return [x1, y1, x2, y2]

    def run_yolov8_object_detection(self, image):
        """Run object detection using YOLOv8 with OpenCV DNN."""
        try:
            # Prepare image for inference
            input_height = 640
            input_width = 640
            
            # Create blob from image
            blob = cv2.dnn.blobFromImage(
                image, 
                1/255.0,  # scaling
                (input_width, input_height),  # size
                swapRB=True,  # BGR to RGB
                crop=False
            )
            
            # Set input and run inference
            self.model.setInput(blob)
            outputs = self.model.forward()
            
            # Process detections
            detected_objects = []
            
            # Reshape output to [num_detections, 85] format
            outputs = outputs[0].transpose((1, 0))
            
            # Filter detections
            for detection in outputs:
                confidence = detection[4]
                
                if confidence > 0.4:  # Confidence threshold
                    class_scores = detection[5:]
                    class_id = np.argmax(class_scores)
                    
                    if class_id == 11:  # Stop sign class
                        # Extract bounding box
                        x = detection[0]
                        y = detection[1]
                        w = detection[2]
                        h = detection[3]
                        
                        # Convert to XYWH format
                        xywh = np.array([x, y, w, h])
                        
                        # Convert to ABCD format
                        abcd = self.xywh2abcd(xywh, image.shape)
                        
                        detected_objects.append({
                            'bounding_box': abcd,
                            'confidence': float(confidence),
                            'class_label': 11
                        })
            
            return detected_objects

        except Exception as e:
            self.get_logger().error(f"Model inference failed: {e}")
            return []

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
            # Ensure the annotated image is in BGR format
            if annotated_image.shape[2] == 4:  # If BGRA
                annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGRA2BGR)

            # Publish YOLO annotated image
            yolo_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding="bgr8")
            self.yolo_pub.publish(yolo_msg)
            self.get_logger().debug("Published YOLO image with bounding boxes.")  # Changed to debug

            # Publish stop flag
            stop_msg = Bool()
            stop_msg.data = stop_flag
            self.stop_pub.publish(stop_msg)
            self.get_logger().debug(f"Published stop flag: {stop_flag}")  # Changed to debug
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