#echo "Entering virtual env \"one_env\""
#source one_env/local/bin/activate
#echo "Entered one_env"
colcon build --symlink-install --packages-select reactive_racing 
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zed2 & ros2 launch reactive_racing race.launch.py
