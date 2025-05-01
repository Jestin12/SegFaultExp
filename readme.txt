/opt/ros/humble/share/turtlebot3_navigation2
ros2 run camera_ros camera_node --ros-args -p format:=BGR888 -p width:=640 -p height:=480
ros2 launch turtlebot3_bringup robot.launch.py

# to ssh to Jestin's Hotspot
ssh ubuntu@192.168.248.9

#for nav2 to put in .bashrc
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

