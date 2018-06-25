import numpy as np
import math
import IPython
import PyKDL
import sys
import random
from scipy.spatial import Delaunay

import rospy
import rospkg
import baxter_interface
from baxter_kdl.kdl_parser import kdl_tree_from_urdf_model
from urdf_parser_py.urdf import URDF
import moveit_commander
import geometry_msgs.msg
from gazebo_msgs.srv import SpawnModel, DeleteModel, GetModelState
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from std_msgs.msg import String

from moveit_python import *
import moveit_python
from utils import InfoGetter, GLI, find_neighbors

# rospy.init_node('baxter_hug')

class Baxter(object):
    def __init__(self, use_moveit=False):
        self.use_moveit = use_moveit
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
        if not self._init_state:
            self._rs.enable()


        if self.use_moveit:
            # moveit group setup
            moveit_commander.roscpp_initialize(sys.argv)
            self.robot = moveit_commander.RobotCommander()
            self.scene = moveit_python.PlanningSceneInterface(self.robot.get_planning_frame())
            # self.scene = moveit_commander.PlanningSceneInterface()
            self.group_name = "right_arm"
            self.group = moveit_commander.MoveGroupCommander(self.group_name)

        # Hugging target
        self.cylinder_height = 1.8
        self.cylinder_radius = 0.1
        self.cylinder1 = np.asarray([0.4, 0.0, -1.0])
        self.cylinder2 = np.asarray([0.4, 0.0, -1.0 + self.cylinder_height])
        segment = 10
        self.target_line = np.empty([segment, 3], float)
        for i in range(segment):
            self.target_line[i] = self.cylinder1 + (self.cylinder2 - self.cylinder1) * i

        # Build line point graph for interaction mesh
        if not self.use_moveit:
            # Joint position control
            self.right_limb_interface.move_to_neutral(timeout=10.0)
        else:
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
        right_limb_pose, _ = limbPose(self.kdl_tree, self.base_link, self.right_limb_interface, 'right')
        self.graph_points = np.concatenate((right_limb_pose[5:], self.target_line), 0)
        self.triangulation = Delaunay(self.graph_points)
        # IPython.embed()


        # Listen to collision information
        # self.collision_getter = InfoGetter()
        # self.collision_topic = "/hug_collision"



    def reset(self):
        print "Resetting Baxter..."
        # limb = 'right'
        # limb_interface = baxter_interface.Limb(limb)

        if not self.use_moveit:
            # Joint position control
            self.right_limb_interface.move_to_neutral(timeout=10.0)
        else:
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

        # ###################### Reset hugging target ##########################
        self.delete_model("hugging_target")
        rospy.sleep(0.1)
        # Randomly initialize target position
        cylinder_x = random.uniform(0.3, 0.7)
        cylinder_y = random.uniform(-0.2, 0.2)
        cylinder_z = (self.cylinder2[2] - self.cylinder1[2]) / 2.0
        self.cylinder1 = (cylinder_x, cylinder_y, -1.0)
        self.cylinder2 = (cylinder_x, cylinder_y, -1.0 + self.cylinder2[2] - self.cylinder1[2])

        print "load gazebo model"
        resp = self.load_model("hugging_target", "cylinder.sdf",
                           Pose(position=Point(x=cylinder_x, y=cylinder_y, z=cylinder_z)))
        # Listen to collision information
        # rospy.Subscriber(self.collision_topic, String, self.collision_getter)

        if self.use_moveit:
            # Delete object in the scene
            count = 0
            while 'target' in self.scene.getKnownCollisionObjects():
                count += 1
                if count > 10:
                    self.scene._collision = []
                self.scene.removeCollisionObject('target', wait=True)
                rospy.sleep(0.1)
                print "deleting target...",

            cylinder_name = "target"
            cylinder_height = self.cylinder2[2] - self.cylinder1[2]
            # Add object to the planning scene for collision avoidance
            while 'target' not in self.scene.getKnownCollisionObjects():
                self.scene.addCylinder(cylinder_name, cylinder_height, self.cylinder_radius,
                                       self.cylinder1[0], self.cylinder1[1], self.cylinder1[2] + cylinder_height / 2.0)
                rospy.sleep(0.1)
                print "adding target..."
            print "cylinder_x: %f, cylinder_y: %f" % (cylinder_x, cylinder_y),

        print "done"



    def reward_evaluation(self, w_last, step):

        # Calculate writhe improvement
        rospy.sleep(0.01)
        limb = 'right'
        right_pose, _ = limbPose(self.kdl_tree, self.base_link, self.right_limb_interface, limb)
        writhe = np.empty((len(self.target_line) - 1, 7))
        for idx, segment in enumerate(self.target_line[:-1]):
            for idx_robot in range(5, 12):
                writhe[idx, idx_robot - 5] = GLI(self.target_line[idx], self.target_line[idx + 1],
                                                 right_pose[idx_robot], right_pose[idx_robot + 1])[0]
        w = np.abs(writhe.flatten().sum())
        reward = (w - w_last) * 100

        # Detect collision
        collision = 0
        g_get_state = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)
        rospy.wait_for_service("/gazebo/get_model_state")
        try:
            state = g_get_state(model_name="hugging_target")
        except Exception, e:
            rospy.logerr('Error on calling service: %s', str(e))
        current_cylinder_pos = state.pose.position
        cylinder_move = math.hypot((current_cylinder_pos.x - self.cylinder2[0]),
                             (current_cylinder_pos.y - self.cylinder2[1]))
        if cylinder_move > 0.03:
            collision = 1
            reward = -1.0
            if step <= 2:
                collision = -1

        # Listen to collision information
        # msg = self.collision_getter.get_msg()
        # print("Collision massage:", msg)
        # if msg:
        #     if msg.data == "cylinder_collision":
        #         collision = 1

        return reward, w, collision


    def getstate(self):

        right_pose, right_joint_pos = limbPose(self.kdl_tree, self.base_link, self.right_limb_interface, 'right')
        left_pose, left_joint_pos = limbPose(self.kdl_tree, self.base_link, self.left_limb_interface, 'left')
        right_joint = [right_joint_pos[0], right_joint_pos[1], right_joint_pos[2], right_joint_pos[3], right_joint_pos[4], right_joint_pos[5], right_joint_pos[6]]

        # right limb joint positions
        state1 = np.asarray(right_joint)

        # right limb link cartesian positions
        state2 = np.asarray(right_pose[3:]).flatten()

        # hugging target -- cylinder
        aa = np.asarray([self.cylinder_radius])
        bb = np.asarray([self.cylinder1, self.cylinder2]).flatten()
        state3 = np.concatenate((aa, bb), axis=0)

        # writhe matrix
        writhe = np.empty((len(self.target_line)-1, 7))
        for idx, segment in enumerate(self.target_line[:-1]):
            for idx_robot in range(5, 12):
                writhe[idx, idx_robot-5] = GLI(self.target_line[idx], self.target_line[idx+1],
                                             right_pose[idx_robot], right_pose[idx_robot+1])[0]
        state4 = writhe.flatten()
        # state4 = np.asarray([GLI(self.cylinder1, self.cylinder2, right_pose[5], right_pose[6])[0],
        #                  GLI(self.cylinder1, self.cylinder2, right_pose[6], right_pose[7])[0],
        #                  GLI(self.cylinder1, self.cylinder2, right_pose[7], right_pose[8])[0],
        #                  GLI(self.cylinder1, self.cylinder2, right_pose[8], right_pose[9])[0],
        #                  GLI(self.cylinder1, self.cylinder2, right_pose[9], right_pose[10])[0],
        #                  GLI(self.cylinder1, self.cylinder2, right_pose[10], right_pose[11])[0],
        #                  GLI(self.cylinder1, self.cylinder2, right_pose[11], right_pose[12])[0]]).flatten()

        # interaction mesh
        InterMesh = np.empty(self.graph_points.shape)
        for idx, point in enumerate(self.graph_points):
            neighbor_index = find_neighbors(idx, self.triangulation)
            W = 0
            Lap = point
            # calculate normalization constant
            for nei_point in self.graph_points[neighbor_index]:
                W = W + 1.0 / math.sqrt( (nei_point[0] - point[0])**2 + (nei_point[1] - point[1])**2 + (nei_point[2] - point[2])**2 )
            # calculate Laplace coordinates
            for nei_point in self.graph_points[neighbor_index]:
                dis_nei = math.sqrt( (nei_point[0] - point[0])**2 + (nei_point[1] - point[1])**2 + (nei_point[2] - point[2])**2 )
                Lap = Lap - nei_point / ( dis_nei * W )
            InterMesh[idx] = Lap
        state5 = InterMesh.flatten()

        state = [state1, state2, state3, state4, state5]
        return state

    def act(self, action):

        # ########### Joint torque control ####################
        # limb_interface.set_joint_torques(cmd)

        if not self.use_moveit:
            cmd = dict()
            for i, joint in enumerate(self.right_limb_interface.joint_names()):
                cmd[joint] = 0.2 * action[i]
            # ########## delta Joint position control ###############
            cur_type_values = self.right_limb_interface.joint_angles()
            for i, joint in enumerate(self.right_limb_interface.joint_names()):
                cmd[joint] = cmd[joint] + cur_type_values[joint]
            try:
                self.right_limb_interface.move_to_joint_positions(cmd, timeout=2.0)
            except Exception, e:
                rospy.logerr('Error: %s', str(e))
        else:
            # ########## moveit joint position move #####################
            joint_goal = self.group.get_current_joint_values()
            for i in range(7):
                joint_goal[i] = joint_goal[i] + 0.2 * action[i]
            try:
                self.group.go(joint_goal, wait=True)
            except Exception, e:
                rospy.logerr('Error: %s', str(e))

            # Calling ``stop()`` ensures that there is no residual movement
            self.group.stop()


    def load_model(self, name, path, block_pose,
                    block_reference_frame="world"):
        # Get Models' Path
        model_path = "./gazebo_models/"

        # Load Block SDF
        block_xml = ''
        with open(model_path + path, "r") as block_file:
            block_xml = block_file.read().replace('\n', '')

        # Spawn Block SDF
        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        resp_sdf = 0
        try:
            spawn_sdf = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
            resp_sdf = spawn_sdf(name, block_xml, "/",
                                   block_pose, block_reference_frame)
        except rospy.ServiceException, e:
            rospy.logerr("Spawn SDF service call failed: {0}".format(e))

        return resp_sdf

    def delete_model(self, name):
        try:
            delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
            resp_delete = delete_model(name)
        except rospy.ServiceException, e:
            rospy.loginfo("Delete Model service call failed: {0}".format(e))


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
        limb_pose.append( [pos[0], pos[1], pos[2]] )

    return np.asarray(limb_pose), kdl_array









