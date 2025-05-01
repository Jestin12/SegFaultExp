/opt/ros/humble/share/turtlebot3_navigation2
ros2 run camera_ros camera_node --ros-args -p format:=BGR888 -p width:=640 -p height:=480
ros2 launch turtlebot3_bringup robot.launch.py

# to ssh to Jestin's Hotspot
ssh ubuntu@192.168.248.9

#for nav2 to put in .bashrc
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp


#launching nav2
ros2 launch turtlebot3_navigation2 navigation2.launch.py use_sim_time:=False slam:=True


#teleop
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r cmd_vel:=/cmd_vel

#copying files to turtlebot
scp -r SegFaultExp/A3/pathfinder/src/drive/ ubuntu@10.42.0.1:turtlebot3_ws/src/turtlebot3

turtlebot3_ws/src/turtlebot3/drive
