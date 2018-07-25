# robot-manipulation-DRL
Address robot manipulation problem using deep reinforcement learning

gazebo-plugins:
    (seems useless)
    git clone https://github.com/roboticsgroup/roboticsgroup_gazebo_plugins.git
    catkin_make --pkg roboticsgroup_gazebo_plugins
for indigo:
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

