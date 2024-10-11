# reactive_racing

_This ROS2 package provides a template for building a racing pipeline_

## Information

The package includes two template nodes:
1. `perceive_boundaries.py` shows how to subscribe to the laser scan from the lidar and publish a point cloud
2. `compute_control.py` shows how to instantiate a timer which runs at a fixed loop and publish steering and velocity commands to the vesc

The package includes an example launch file `race.launch.py` which launches the following nodes:
1. LIDAR: `urg_node>urg_node_driver`
2. Speed Controller:  `vesc_driver>vesc_driver_node`
3. Ackermann Commands to Speed Controller Values: `vesc_ackermann>ackermann_to_vesc_node`
4. Multiplexing Ackermann Command Sources: `ackermann_mux>ackermann_mux`
5. Template Perception Node: `reactive_racing>perception`
6. Template Control Node: `reactive_racing>control`

## Building

1. `cd ~/colcon_ws/`
2. `colcon build` to build the whole workspace or `colcon build --packages-select reactive_racing` to build just this package

## Running

1. `ros2 launch reactive_racing race.launch.py`
