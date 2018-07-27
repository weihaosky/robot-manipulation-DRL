# robot-manipulation-DRL
Address robot manipulation problem using deep reinforcement learning


1. Run simulation environment:
```
cd catkin_ws
. ./baxter.sh sim
roslaunch baxter_gazebo baxter_world.launch
```

2. load humanoid medel:
```
roslaunch baxter_hug humanoid.launch 
rosrun rqt_gui rqt_gui
publish to topic "/humanoid/left_joint_position_controller/command" 0.125*sin(i/200)+0.125
```

Requirements:

gazebo-plugins:
    (seems useless)
```
git clone https://github.com/roboticsgroup/roboticsgroup_gazebo_plugins.git
catkin_make --pkg roboticsgroup_gazebo_plugins
```
for indigo:
```
git clone https://github.com/ros-controls/ros_controllers.git
git checkout indigo-devel

modify joint_position_controller.cpp:
    if (!urdf.initParam("humanoid/robot_description"))
    {
        if (!urdf.initParam("robot_description"))
        {
        ROS_ERROR("Failed to parse urdf file");
        return false;
        }
    }
catkin_make --pkg velocity_controllers
```

