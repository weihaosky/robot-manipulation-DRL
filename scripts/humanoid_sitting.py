import rospy
import rospkg, roslaunch
import baxter_interface
from baxter_kdl.kdl_parser import kdl_tree_from_urdf_model
from urdf_parser_py.urdf import URDF
import moveit_commander
import geometry_msgs.msg
from gazebo_msgs.srv import SpawnModel, DeleteModel, GetModelState, GetLinkState, SetModelState, SetModelConfiguration
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from std_msgs.msg import String
import tf
import IPython

def main():
    set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
    set_model_config = rospy.ServiceProxy('/gazebo/set_model_configuration', SetModelConfiguration)
    for i in range(10000):
        model_msg = ModelState()
        model_msg.model_name = "humanoid"
        model_msg.reference_frame = "world"
        model_msg.pose.position.x = 0.6
        model_msg.pose.position.y = 0.0
        model_msg.pose.position.z = -0.4
        resp_set = set_model_state(model_msg)

    # set_model_config('baxter', 'robot_description', ['pedestal_fixed'], [-0.3])

    # for i in range(1000):
    #     height = -0.4 + i / 1000.0
    #     set_model_config('baxter', 'robot_description', ['pedestal_fixed'], [height])
    #     rospy.sleep(0.01)


if __name__ == '__main__':
    main()