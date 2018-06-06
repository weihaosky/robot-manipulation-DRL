import numpy as np
import math
from scipy.integrate import quad,dblquad,nquad
import IPython
import PyKDL

import rospy
import baxter_interface
from baxter_kdl.kdl_parser import kdl_tree_from_urdf_model
from urdf_parser_py.urdf import URDF

# rospy.init_node('baxter_hug')

class Baxter(object):
    def __init__(self):
        self.baxter = URDF.from_parameter_server(key='robot_description')
        self.kdl_tree = kdl_tree_from_urdf_model(self.baxter)
        self.base_link = self.baxter.get_root()

        self.right_limb_interface = baxter_interface.Limb('right')
        self.left_limb_interface = baxter_interface.Limb('left')

        # Verify robot is enabled
        print("Getting robot state... ")
        self._rs = baxter_interface.RobotEnable()
        self._init_state = self._rs.state().enabled
        print("Enabling robot... ")
        self._rs.enable()

    def reset(self):
        print "Resetting Baxter...",
        # limb = 'right'
        # limb_interface = baxter_interface.Limb(limb)

        # Joint position control
        self.right_limb_interface.move_to_neutral(timeout=10.0)

        print "done"



    def reward_evaluation(self, w_last):
        cylinder1 = (0.5, 0.0, -1.0)
        cylinder2 = (0.5, 0.0, 0.5)

        limb = 'right'
        limb_pose, _ = limbPose(self.kdl_tree, self.base_link, self.right_limb_interface, limb)
        w = GLI(cylinder1, cylinder2, limb_pose[5], limb_pose[7])[0] + \
            GLI(cylinder1, cylinder2, limb_pose[7], limb_pose[8])[0] + \
            GLI(cylinder1, cylinder2, limb_pose[8], limb_pose[9])[0]
        w = np.abs(w)

        reward = w - w_last

        return reward, w


    def getstate(self):
        right_pose, right_joint_pos = limbPose(self.kdl_tree, self.base_link, self.right_limb_interface, 'right')
        left_pose, left_joint_pos = limbPose(self.kdl_tree, self.base_link, self.left_limb_interface, 'left')
        right_joint = [right_joint_pos[0], right_joint_pos[1], right_joint_pos[2], right_joint_pos[3], right_joint_pos[4], right_joint_pos[5], right_joint_pos[6]]
        state1 = np.asarray(right_joint)
        state2 = np.asarray(right_pose[3:]).flatten()
        state = [state1, state2]
        return state

    def act(self, action):

        # limb = 'right'
        # limb_interface = baxter_interface.Limb(limb)
        cmd = dict()
        for i, joint in enumerate(self.right_limb_interface.joint_names()):
            cmd[joint] = action[i]

        # Joint torque control
        # limb_interface.set_joint_torques(cmd)

        # Joint position control
        try:
            self.right_limb_interface.move_to_joint_positions(cmd, timeout=2.0)
        except Exception, e:
            rospy.logerr('Error: %s', str(e))




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
        #IPython.embed()
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




