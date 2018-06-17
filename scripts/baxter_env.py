import numpy as np
import math
from scipy.integrate import quad,dblquad,nquad
import IPython
import PyKDL

import rospy
import baxter_interface
from baxter_kdl.kdl_parser import kdl_tree_from_urdf_model
from urdf_parser_py.urdf import URDF
import moveit_commander
import sys
import geometry_msgs.msg
import random

from moveit_python import *
import moveit_python

# rospy.init_node('baxter_hug')

class Baxter(object):
    def __init__(self):
        self.baxter = URDF.from_parameter_server(key='robot_description')
        self.kdl_tree = kdl_tree_from_urdf_model(self.baxter)
        self.base_link = self.baxter.get_root()

        self.right_limb_interface = baxter_interface.Limb('right')
        self.left_limb_interface = baxter_interface.Limb('left')

        # moveit group setup
        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_python.PlanningSceneInterface(self.robot.get_planning_frame())
        # self.scene = moveit_commander.PlanningSceneInterface()
        self.group_name = "right_arm"
        self.group = moveit_commander.MoveGroupCommander(self.group_name)

        # Hugging target
        self.cylinder1 = (0.4, 0.0, -1.0)
        self.cylinder2 = (0.4, 0.0, 0.5)
        # cylinder_pose = geometry_msgs.msg.PoseStamped()
        # cylinder_pose.header.frame_id = self.robot.get_planning_frame()
        # cylinder_pose.pose.orientation.x = 0.4
        # cylinder_pose.pose.orientation.y = 0.0


        # rospy.sleep(2)
        # box_pose = geometry_msgs.msg.PoseStamped()
        # box_pose.header.frame_id = self.robot.get_planning_frame()
        # box_pose.pose.orientation.w = 1.0
        # box_name = "box"
        # self.scene.add_box(box_name, box_pose, size=(0.1, 0.1, 0.1))
        # self.scene.get_known_object_names()

        # Verify robot is enabled
        print("Getting robot state... ")
        self._rs = baxter_interface.RobotEnable()
        self._init_state = self._rs.state().enabled
        print("Enabling robot... ")
        if not self._init_state:
            self._rs.enable()


    def reset(self):
        print "Resetting Baxter...",
        # limb = 'right'
        # limb_interface = baxter_interface.Limb(limb)

        # Joint position control
        # self.right_limb_interface.move_to_neutral(timeout=10.0)

        # Moveit joint move
        joint_goal = self.group.get_current_joint_values()
        joint_goal[0] = 0.0
        joint_goal[1] = -0.55
        joint_goal[2] = 0.0
        joint_goal[3] = 0.75
        joint_goal[4] = 0.0
        joint_goal[5] = 1.26
        joint_goal[6] = 0.0
        self.group.go(joint_goal, wait=True)
        self.group.stop()

        # Reset hugging target
        count = 0
        while 'target' in self.scene.getKnownCollisionObjects():
            count+=1
            if count > 10:
                self.scene._collision = []
            self.scene.removeCollisionObject('target', wait=True)
            rospy.sleep(0.1)
            print "deleting target...",
        # Randomly initialize target position
        cylinder_x = random.uniform(0.3, 0.7)
        cylinder_y = random.uniform(-0.2, 0.2)
        self.cylinder1 = (cylinder_x, cylinder_y, -1.0)
        self.cylinder2 = (cylinder_x, cylinder_y, 0.5)

        cylinder_name = "target"
        cylinder_height = self.cylinder2[2] - self.cylinder1[2]
        cylinder_radius = 0.1
        while 'target' not in self.scene.getKnownCollisionObjects():
            self.scene.addCylinder(cylinder_name, cylinder_height, cylinder_radius,
                                   self.cylinder1[0], self.cylinder1[1], self.cylinder1[2] + cylinder_height / 2.0)
            rospy.sleep(0.1)
            print "adding target..."
        print "cylinder_x: %f, cylinder_y: %f" % (cylinder_x, cylinder_y),

        print "done"



    def reward_evaluation(self, w_last):

        limb = 'right'
        rospy.sleep(0.01)
        limb_pose, _ = limbPose(self.kdl_tree, self.base_link, self.right_limb_interface, limb)

        w = GLI(self.cylinder1, self.cylinder2, limb_pose[5], limb_pose[7])[0] + \
            GLI(self.cylinder1, self.cylinder2, limb_pose[7], limb_pose[8])[0] + \
            GLI(self.cylinder1, self.cylinder2, limb_pose[8], limb_pose[9])[0]
        w = np.abs(w)

        reward = (w - w_last) * 100

        return reward, w


    def getstate(self):

        right_pose, right_joint_pos = limbPose(self.kdl_tree, self.base_link, self.right_limb_interface, 'right')
        left_pose, left_joint_pos = limbPose(self.kdl_tree, self.base_link, self.left_limb_interface, 'left')
        right_joint = [right_joint_pos[0], right_joint_pos[1], right_joint_pos[2], right_joint_pos[3], right_joint_pos[4], right_joint_pos[5], right_joint_pos[6]]

        # right limb joint positions
        state1 = np.asarray(right_joint)

        # right limb link cartesian positions
        state2 = np.asarray(right_pose[3:]).flatten()

        # hugging target -- cylinder
        state3 = np.asarray([self.cylinder1, self.cylinder2]).flatten()

        # writhe matrix
        state4 = np.asarray([GLI(self.cylinder1, self.cylinder2, right_pose[5], right_pose[7])[0], \
                            GLI(self.cylinder1, self.cylinder2, right_pose[7], right_pose[8])[0], \
                            GLI(self.cylinder1, self.cylinder2, right_pose[8], right_pose[9])[0]]).flatten()

        state = [state1, state2, state3, state4]
        return state

    def act(self, action):

        # limb = 'right'
        # limb_interface = baxter_interface.Limb(limb)
        cmd = dict()
        for i, joint in enumerate(self.right_limb_interface.joint_names()):
            cmd[joint] = 0.2 * action[i]

        # ########### Joint torque control ####################
        # limb_interface.set_joint_torques(cmd)

        # ########## delta Joint position control ###############
        # cur_type_values = self.right_limb_interface.joint_angles()
        # for i, joint in enumerate(self.right_limb_interface.joint_names()):
        #     cmd[joint] = cmd[joint] + cur_type_values[joint]
        # try:
        #     self.right_limb_interface.move_to_joint_positions(cmd, timeout=2.0)
        # except Exception, e:
        #     rospy.logerr('Error: %s', str(e))

        # ########## moveit joint position move #####################
        joint_goal = self.group.get_current_joint_values()
        for i in range(7):
            joint_goal[i] = joint_goal[i] + 0.2 * action[i]
        try:
            self.group.go(joint_goal, wait=True)
        except Exception, e:
            rospy.logerr('Error: %s', str(e))
            #IPython.embed()

        # Calling ``stop()`` ensures that there is no residual movement
        self.group.stop()


def limbPose(kdl_tree, base_link, limb_interface, limb = 'right'):
    tip_link = limb + '_gripper'
    tip_frame = PyKDL.Frame()
    arm_chain = kdl_tree.getChain(base_link, tip_link)

    # Baxter Interface Limb Instances
    #limb_interface = baxter_interface.Limb(limb)
    joint_names = limb_interface.joint_names()
    num_jnts = len(joint_names)

    if limb == 'right':
        limb_link = ['base', 'torso', 'right_arm_mount', 'right_upper_shoulder', 'right_lower_shoulder',
                      'right_upper_elbow', 'right_lower_elbow', 'right_upper_forearm', 'right_lower_forearm',
                      'right_wrist', 'right_hand', 'right_gripper_base', 'right_gripper']
    else:
        limb_link = ['base', 'torso', 'left_arm_mount', 'left_upper_shoulder', 'left_lower_shoulder',
                     'left_upper_elbow', 'left_lower_elbow', 'left_upper_forearm', 'left_lower_forearm',
                     'left_wrist', 'left_hand', 'left_gripper_base', 'left_gripper']
    limb_frame = []
    limb_chain = []
    limb_pose = []
    limb_fk = []

    for idx in xrange(arm_chain.getNrOfSegments()):
        linkname = limb_link[idx]
        limb_frame.append(PyKDL.Frame())
        limb_chain.append(kdl_tree.getChain(base_link, linkname))
        limb_fk.append(PyKDL.ChainFkSolverPos_recursive(kdl_tree.getChain(base_link, linkname)))

    # get the joint positions
    cur_type_values = limb_interface.joint_angles()
    while len(cur_type_values) != 7:
        print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        cur_type_values = limb_interface.joint_angles()
    kdl_array = PyKDL.JntArray(num_jnts)
    for idx, name in enumerate(joint_names):
        kdl_array[idx] = cur_type_values[name]

    limb_joint = [PyKDL.JntArray(1),
                   PyKDL.JntArray(2),
                   PyKDL.JntArray(3),
                   PyKDL.JntArray(4),
                   PyKDL.JntArray(5),
                   PyKDL.JntArray(6),
                   PyKDL.JntArray(7)]
    for i in range(7):
        for j in range(i+1):
            # print i, j
            limb_joint[i][j] = kdl_array[j]


    for i in range(arm_chain.getNrOfSegments()):
        joint_array = limb_joint[limb_chain[i].getNrOfJoints()-1]
        limb_fk[i].JntToCart(joint_array,  limb_frame[i])
        pos = limb_frame[i].p
        rot = PyKDL.Rotation(limb_frame[i].M)
        rot = rot.GetQuaternion()
        limb_pose.append( np.array([pos[0], pos[1], pos[2]]) )

    return limb_pose, kdl_array





# calculate gussian linking integral for two lines X1--X2 and Y1--Y2
def GLI(X1=(0.0, 0.0, 0.0), X2 = (1.0, 0.0, 1.0), Y1 = (0.0, 0.0, 1.0), Y2 = (0.0, 1.0, 1.0)):
    x1 = X1[0]
    y1 = X1[1]
    z1 = X1[2]
    x2 = X2[0]
    y2 = X2[1]
    z2 = X2[2]

    a1 = Y1[0]
    b1 = Y1[1]
    c1 = Y1[2]
    a2 = Y2[0]
    b2 = Y2[1]
    c2 = Y2[2]

    D1 = np.array([x2-x1, y2-y1, z2-z1])
    D2 = np.array([a2-a1, b2-b1, c2-c1])
    Prod = np.cross(D1, D2)

    def inte(s, t):
        x = x1 + (x2 - x1) * s
        y = y1 + (y2 - y1) * s
        z = z1 + (z2 - z1) * s
        a = a1 + (a2 - a1) * t
        b = b1 + (b2 - b1) * t
        c = c1 + (c2 - c1) * t
        return (Prod[0]*(x-a) + Prod[1]*(y-b) + Prod[2]*(z-c)) / np.power((x-a)**2 + (y-b)**2 + (z-c)**2, 3.0/2)

    result = dblquad( inte , 0, 1, lambda t:0, lambda t:1)
    gli = result[0] / (4*math.pi)
    error = result[1] / (4*math.pi)
    return (gli, error)




