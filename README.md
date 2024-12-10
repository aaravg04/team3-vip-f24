# F1Tenth Autonomous Racing Project
Team members: Aarav Gupta, Edmund Shieh, Mehul Rao
---
This repository documents the setup and implementation of an **F1-Tenth autonomous racing car** using computer vision for stop sign detection and the Gap Disparity Extender algorithm for navigation. Follow the detailed setup instructions below to configure your hardware and software. The project aims to create a robust and efficient autonomous racing system optimized for the F1Tenth platform. We took inspiration from https://github.com/ForzaETH/race_stack.

Final demos:
- Speed run: https://youtube.com/shorts/ftj6eM1NPwI?feature=share
- Stop sign detection: https://youtube.com/shorts/NyrI_2MYtWM

![image](https://github.com/user-attachments/assets/88c08c07-54cc-4922-abbb-212ede254631)

![image](https://github.com/user-attachments/assets/c3087a85-9675-4ede-ae11-3e90a83c7af7)


---

## Table of Contents

1. [Stop Sign Detection Using Computer Vision](#stop-sign-detection-using-computer-vision)
2. [Gap Disparity Extender Algorithm for Navigation](#gap-disparity-extender-algorithm-for-navigation)
3. [Hardware and Software Summary](#summary-of-hardware-and-software)

<img src="https://github.com/user-attachments/assets/186fd917-efbb-443f-96f9-23f0efff1706" alt="462579635_2107265463093316_100108676632149157_n" width="500"/>

---

## Stop Sign Detection Using Computer Vision

### Methodology

#### Objective
Use computer vision to detect stop signs and control the vehicle's behavior.

#### Algorithm 
1. Capture camera frames using the ZED2 stereo camera. 
2. Preprocess frames (resize, normalize, and filter) for detection.
3. Use a pretrained yolov11 model to ensure high detection accuracy.
4. Implement post-processing for bounding box extraction and confidence thresholding.
5. When a stop sign is detected:
   - Halt the vehicle for a predefined duration.
   - Resume navigation.
  <img src="https://github.com/user-attachments/assets/693fcadb-d01f-4ba2-9cc4-989d4ab08765" alt="462581421_988892813264572_139218327184769926_n" width="300"/>


#### Implementation
- The vision node runs on the Jetson Orin Nano using **OpenCV** and **TensorRT** for accelerated inference.
- Models are optimized for low latency using **NVIDIA TensorRT**.

## Gap Disparity Extender Algorithm for Navigation

### Methodology

#### Objective
Navigate efficiently and avoid obstacles based on LiDAR data.

#### Algorithm
1. Process LiDAR scan data to identify all gaps in the environment.
2. Select the largest navigable gap.
3. Extend the gap disparity method by dynamically adjusting the trajectory for obstacle clearance.
4. The algorithm ensures smooth and safe navigation at high speeds.

#### Implementation
- The algorithm is implemented as a **ROS 2 node**.
- **Inputs**: Raw LiDAR scans, vehicle speed, and trajectory constraints.
- **Outputs**: Steering and throttle commands.
- Integrates with **Ackermann control** for smooth vehicle operation.

## Summary of Hardware and Software

### Hardware
1. **Jetson Orin Nano Developer Kit**
   - Onboard computation for running vision and navigation algorithms.
2. **Hokuyo UTM-30LX LiDAR**
   - Provides high-resolution distance measurements for obstacle detection and navigation.
3. **ZED2 Stereo Camera**
   - Captures frames for computer vision tasks such as stop sign detection.
4. **VESC (Vehicle Electric Speed Controller)**
   - Controls the motor and steering of the vehicle.
5. **USB Hub**
   - Connects peripheral devices like the ZED2 camera to the Jetson.
6. **NVME SSD**
   - Provides high-speed storage for the Jetson.

### Software
1. **Ubuntu 20.04/22.04**
   - Operating system for the host computer and Jetson Orin Nano.
2. **NVIDIA Jetpack 6.0**
   - Drivers and SDK for GPU acceleration on the Jetson.
3. **ROS 2 Humble**
   - Framework for building and running robotic software.
4. **OpenCV**
   - Library for image processing used in stop sign detection.
5. **TensorRT**
   - NVIDIA's inference optimizer for deep learning models.
6. **F1-Tenth System Repositories**
   - Includes software for integrating LIDAR, camera, and control systems.
7. **ZED SDK**
   - Driver and software package for interfacing with the ZED2 stereo camera.




