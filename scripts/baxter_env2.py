import numpy as np
from numpy import asarray
import math
import IPython
import PyKDL
import sys
import random
from scipy.spatial import Delaunay
import multiprocessing as mp
import threading
import time

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

from moveit_python import *
import moveit_python
from utils import InfoGetter, GLI, find_neighbors


class Baxter(object):
    def __init__(self, use_moveit=False):
        self.use_moveit = use_moveit
        self.baxter = URDF.from_parameter_server(key='robot_description')
        self.kdl_tree = kdl_tree_from_urdf_model(self.baxter)
        self.base_link = self.baxter.get_root()

        self.right_limb_interface = baxter_interface.Limb('right')
        self.left_limb_interface = baxter_interface.Limb('left')

        self.get_link_state = rospy.ServiceProxy("/gazebo/get_link_state", GetLinkState)
        self.get_model_state = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)
        self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.set_model_config = rospy.ServiceProxy('/gazebo/set_model_configuration', SetModelConfiguration)

        # Listen to collision information
        # self.collision_getter = InfoGetter()
        # self.collision_topic = "/hug_collision"

        # Verify robot is enabled
        print("Getting robot state... ")
        self._rs = baxter_interface.RobotEnable()
        self._init_state = self._rs.state().enabled
        if not self._init_state:
            print("Enabling robot... ")
            self._rs.enable()

        if self.use_moveit:
            # Moveit group setup
            moveit_commander.roscpp_initialize(sys.argv)
            self.robot = moveit_commander.RobotCommander()
            self.scene = moveit_python.PlanningSceneInterface(self.robot.get_planning_frame())
            # self.scene = moveit_commander.PlanningSceneInterface()
            self.group_name = "right_arm"
            self.group = moveit_commander.MoveGroupCommander(self.group_name)

        # ######################## Create Hugging target #############################
        # humanoid hugging target # remember to make sure target_line correct + - y
        self.target_pos_start = np.asarray([0.5, 0, -0.93]) # robot /base frame, z = -0.93 w.r.t /world frame
        self.target_line_start = np.empty([22, 3], float)
        for i in range(11):
            self.target_line_start[i] = self.target_pos_start + [0, -0.0, 1.8] - (asarray([0, -0.0, 1.8]) - asarray([0, -0.0, 0.5]))/10*i
            self.target_line_start[i+11] = self.target_pos_start + [0, -0.5, 1.3] + (asarray([0, 0.5, 1.3]) - asarray([0, -0.5, 1.3]))/10*i
        self.target_line = self.target_line_start

        # Build line point graph for interaction mesh
        # reset right arm
        start_point_right = [-0.3, 1.0, 0.0, 0.5, 0.0, 0.027, 0.0]
        t01 = threading.Thread(target=resetarm_job, args=(self.right_limb_interface, start_point_right))
        t01.start()
        # reset left arm
        start_point_left = [0.3, 1.0, 0.0, 0.5, 0.0, 0.027, 0.0]
        t02 = threading.Thread(target=resetarm_job, args=(self.left_limb_interface, start_point_left))
        t02.start()
        t02.join()
        t01.join()

        right_limb_pose, _ = limbPose(self.kdl_tree, self.base_link, self.right_limb_interface, 'right')
        left_limb_pose, _ = limbPose(self.kdl_tree, self.base_link, self.left_limb_interface, 'left')
        graph_points = np.concatenate((right_limb_pose[5:], left_limb_pose[5:], self.target_line_start), 0)
        self.triangulation = Delaunay(graph_points)


    def reset(self, episode_num, collision):
        print "Resetting Baxter..."
        self.delete_model("humanoid")
        rospy.sleep(0.1)

        # reset arm position
        # right arm
        start_point_right = [-0.3, 1.0, 0.0, 0.5, 0.0, 0.027, 0.0]
        t01 = threading.Thread(target=resetarm_job, args=(self.right_limb_interface, start_point_right))
        t01.start()
        # left arm
        start_point_left = [0.3, 1.0, 0.0, 0.5, 0.0, 0.027, 0.0]
        t02 = threading.Thread(target=resetarm_job, args=(self.left_limb_interface, start_point_left))
        t02.start()
        t02.join()
        t01.join()

        # ###################### Reset hugging target ##########################
        reset_mode = np.random.choice([1, 3])
        if reset_mode == 1 or reset_mode == 2:
            self.target_line_start = self.target_line_start - self.target_pos_start
            self.target_pos_start[0] = random.uniform(0.4, 0.8)
            self.target_pos_start[1] = random.uniform(-0.2, 0.2)
            self.target_pos_start[2] = 0.0
            self.target_line_start = self.target_line_start + self.target_pos_start
            self.target_line = self.target_line_start
            print "load gazebo model"
            humanoid_pose = Pose()
            humanoid_pose.position.x = self.target_pos_start[0]
            humanoid_pose.position.y = self.target_pos_start[1]
            humanoid_pose.position.z = self.target_pos_start[2]
            resp = self.load_model("humanoid", "humanoid/humanoid-static.urdf", humanoid_pose, type="urdf")
            rospy.sleep(0.1)

        if reset_mode == 3:
            self.target_line_start = self.target_line_start - self.target_pos_start
            self.target_pos_start[0] = 1.7
            self.target_pos_start[1] = random.uniform(-0.2, 0.2)
            self.target_pos_start[2] = random.uniform(1.2, 1.4)
            self.target_line_start = self.target_line_start + self.target_pos_start
            self.target_line = self.target_line_start
            print "load gazebo model"
            humanoid_pose = Pose()
            humanoid_pose.position.x = self.target_pos_start[0]
            humanoid_pose.position.y = self.target_pos_start[1]
            humanoid_pose.position.z = self.target_pos_start[2]
            quaternion0 = tf.transformations.quaternion_from_euler(0.0, -1.57, 0.0)
            humanoid_pose.orientation.x = quaternion0[0]
            humanoid_pose.orientation.y = quaternion0[1]
            humanoid_pose.orientation.z = quaternion0[2]
            humanoid_pose.orientation.w = quaternion0[3]
            resp = self.load_model("humanoid", "humanoid/humanoid-static-withouthead.urdf", humanoid_pose, type="urdf")
            rospy.sleep(0.1)

        if reset_mode == 4:
            self.target_line_start = self.target_line_start - self.target_pos_start
            self.target_pos_start[0] = 0.5
            self.target_pos_start[1] = 1.2
            self.target_pos_start[2] = 1.3
            self.target_line_start = self.target_line_start + self.target_pos_start
            self.target_line = self.target_line_start
            print "load gazebo model"
            humanoid_pose = Pose()
            humanoid_pose.position.x = self.target_pos_start[0]
            humanoid_pose.position.y = self.target_pos_start[1]
            humanoid_pose.position.z = self.target_pos_start[2]
            quaternion0 = tf.transformations.quaternion_from_euler(0.0, -1.57, 1.57)
            humanoid_pose.orientation.x = quaternion0[0]
            humanoid_pose.orientation.y = quaternion0[1]
            humanoid_pose.orientation.z = quaternion0[2]
            humanoid_pose.orientation.w = quaternion0[3]
            resp = self.load_model("humanoid", "humanoid/humanoid-static-left.urdf", humanoid_pose, type="urdf")
            rospy.sleep(0.1)

        if reset_mode == 5:
            self.target_line_start = self.target_line_start - self.target_pos_start
            self.target_pos_start[0] = 0.5
            self.target_pos_start[1] = -1.2
            self.target_pos_start[2] = 1.3
            self.target_line_start = self.target_line_start + self.target_pos_start
            self.target_line = self.target_line_start
            print "load gazebo model"
            humanoid_pose = Pose()
            humanoid_pose.position.x = self.target_pos_start[0]
            humanoid_pose.position.y = self.target_pos_start[1]
            humanoid_pose.position.z = self.target_pos_start[2]
            quaternion0 = tf.transformations.quaternion_from_euler(0.0, -1.57, -1.57)
            humanoid_pose.orientation.x = quaternion0[0]
            humanoid_pose.orientation.y = quaternion0[1]
            humanoid_pose.orientation.z = quaternion0[2]
            humanoid_pose.orientation.w = quaternion0[3]
            resp = self.load_model("humanoid", "humanoid/humanoid-static-right.urdf", humanoid_pose, type="urdf")
            rospy.sleep(0.1)

        print "done"

    def reward_evaluation(self, w_last, step):

        rospy.wait_for_service('/gazebo/get_link_state')
        torso_pose = self.get_link_state("humanoid::Torso_link", "world").link_state.pose
        T = tf.transformations.quaternion_matrix([torso_pose.orientation.x,
                                              torso_pose.orientation.y,
                                              torso_pose.orientation.z,
                                              torso_pose.orientation.w])
        # rotation in /world frame, which is [0, 0, -0.93] to /base frame
        # target_line_start-target_pos_start is the original position of human in /world frame
        self.target_line = np.dot(T[:3,:3], (self.target_line_start-self.target_pos_start).T).T + \
                                   [torso_pose.position.x, torso_pose.position.y, torso_pose.position.z] + \
                                    [0, 0, -0.93]

        # Calculate writhe improvement
        rospy.sleep(0.01)
        right_limb_pose, _ = limbPose(self.kdl_tree, self.base_link, self.right_limb_interface, 'right')
        left_limb_pose, _ = limbPose(self.kdl_tree, self.base_link, self.left_limb_interface, 'left')
        writhe = np.empty((len(self.target_line) - 2, 14))
        for idx_target in range(10):
            for idx_robot in range(5, 12):
                x1_right = self.target_line[idx_target].copy()
                x2_right = self.target_line[idx_target + 1].copy()
                x1_right[1] -= 0.15
                x2_right[1] -= 0.15
                writhe[idx_target, idx_robot - 5] = GLI(x1_right, x2_right,
                                                        right_limb_pose[idx_robot], right_limb_pose[idx_robot + 1])[0]
                x1_left = self.target_line[idx_target].copy()
                x2_left = self.target_line[idx_target + 1].copy()
                x1_left[1] += 0.15
                x2_left[1] += 0.15
                writhe[idx_target, idx_robot - 5 + 7] = GLI(x1_left, x2_left,
                                                            left_limb_pose[idx_robot], left_limb_pose[idx_robot + 1])[0]
                # writhe[idx_target, idx_robot-5] = \
                #     GLI(self.target_line[idx_target], self.target_line[idx_target+1],
                #         right_limb_pose[idx_robot], right_limb_pose[idx_robot+1])[0]
                # writhe[idx_target, idx_robot-5+7] = \
                #     GLI(self.target_line[idx_target], self.target_line[idx_target+1],
                #         left_limb_pose[idx_robot], left_limb_pose[idx_robot+1])[0]
        for idx_target in range(11, 21):
            for idx_robot in range(5, 12):
                writhe[idx_target-1, idx_robot-5] = \
                    GLI(self.target_line[idx_target], self.target_line[idx_target+1],
                        right_limb_pose[idx_robot], right_limb_pose[idx_robot+1])[0]
                writhe[idx_target-1, idx_robot-5+7] = \
                    GLI(self.target_line[idx_target], self.target_line[idx_target+1],
                        left_limb_pose[idx_robot], left_limb_pose[idx_robot+1])[0]
        w_right1 = np.abs(writhe[0:10, 0:7].flatten().sum())
        w_right2 = np.abs(writhe[0:10, 7:14].flatten().sum())
        w_left1 = np.abs(writhe[10:20, 0:7].flatten().sum())
        w_left2 = np.abs(writhe[10:20, 7:14].flatten().sum())
        w = w_right1 + w_right2 + w_left1 + w_left2
        reward = (w - w_last) * 50 - 7.5 + w * 5

        # Prevent robot arms above humanoid arms
        height_human = self.target_line[4][2] + 0.02
        height_robot = (right_limb_pose[5:][:, 2].mean() + left_limb_pose[5:][:, 2].mean())/2.0
        r2 = height_robot - height_human
        print r2
        if r2 > 0:
            print("!!!!!!!!!!!!!!!!!!!!!!!!\n!!!!!!!!!!!!!!!!!!!!!!!!\n!!!!!!!!!!!!!!!!!!!!!!!!")
            reward = reward - r2

        # Detect collision
        collision = 0
        # current_pos = self.target_line[4]
        # target_move = math.hypot((current_pos[0] - self.target_pos_start[0]),
        #                          (current_pos[1] - self.target_pos_start[1]))
        #
        # print("target_move:" , target_move)
        # if target_move > 0.4:
        #     # collision = 1   # collision
        #     collision = 1
        #     # reward = 0.0
        # if target_move > 0.2 and step <= 2:
        #     collision = -1  # model load error

        # Listen to collision information
        # msg = self.collision_getter.get_msg()
        # if msg:
        #     if msg.data == "cylinder_collision":
        #         collision = 1

        return reward, w, collision


    def getstate(self):

        rospy.wait_for_service('/gazebo/get_link_state')
        torso_pose = self.get_link_state("humanoid::Torso_link", "world").link_state.pose
        T = tf.transformations.quaternion_matrix([torso_pose.orientation.x,
                                                  torso_pose.orientation.y,
                                                  torso_pose.orientation.z,
                                                  torso_pose.orientation.w])
        # rotation in /world frame, which is [0, 0, -0.93] to /base frame
        # target_line_start-target_pos_start is the original position of human in /world frame
        self.target_line = np.dot(T[:3, :3], (self.target_line_start - self.target_pos_start).T).T + \
                           [torso_pose.position.x, torso_pose.position.y, torso_pose.position.z] + \
                           [0, 0, -0.93]
        print self.target_line

        right_limb_pose, right_joint_pos = limbPose(self.kdl_tree, self.base_link, self.right_limb_interface, 'right')
        left_limb_pose, left_joint_pos = limbPose(self.kdl_tree, self.base_link, self.left_limb_interface, 'left')
        right_joint = [right_joint_pos[0], right_joint_pos[1], right_joint_pos[2], right_joint_pos[3],
                       right_joint_pos[4], right_joint_pos[5], right_joint_pos[6]]
        left_joint = [left_joint_pos[0], left_joint_pos[1], left_joint_pos[2], left_joint_pos[3],
                       left_joint_pos[4], left_joint_pos[5], left_joint_pos[6]]
        # right limb joint positions
        state1 = np.asarray([right_joint, left_joint]).flatten()

        # right limb link cartesian positions
        state2 = np.asarray([right_limb_pose[5:], left_limb_pose[5:]]).flatten()

        # hugging target -- cylinder
        # aa = np.asarray([self.cylinder_radius])
        # bb = np.asarray([self.cylinder1, self.cylinder1 + asarray([0, 0, self.cylinder_height])]).flatten()
        # state3 = np.concatenate((aa, bb), axis=0)
        state3 = self.target_line.flatten()

        # ####################### writhe matrix ###########################
        writhe = np.empty((len(self.target_line) - 2, 14))
        for idx_target in range(10):
            for idx_robot in range(5, 12):
                writhe[idx_target, idx_robot-5] = \
                    GLI(self.target_line[idx_target], self.target_line[idx_target+1],
                        right_limb_pose[idx_robot], right_limb_pose[idx_robot+1])[0]
                writhe[idx_target, idx_robot-5+7] = \
                    GLI(self.target_line[idx_target], self.target_line[idx_target+1],
                        left_limb_pose[idx_robot], left_limb_pose[idx_robot+1])[0]
        for idx_target in range(11, 21):
            for idx_robot in range(5, 12):
                writhe[idx_target-1, idx_robot-5] = \
                    GLI(self.target_line[idx_target], self.target_line[idx_target+1],
                        right_limb_pose[idx_robot], right_limb_pose[idx_robot+1])[0]
                writhe[idx_target-1, idx_robot-5+7] = \
                    GLI(self.target_line[idx_target], self.target_line[idx_target+1],
                        left_limb_pose[idx_robot], left_limb_pose[idx_robot+1])[0]
        state4 = writhe.flatten()

        # ############################### interaction mesh ##################################
        graph_points = np.concatenate((right_limb_pose[5:], left_limb_pose[5:], self.target_line), 0)
        InterMesh = np.empty(graph_points.shape)
        mesh1 = []
        mesh2 = []
        for idx, point in enumerate(graph_points):
            neighbor_index = find_neighbors(idx, self.triangulation)
            W = 0
            Lap = point
            # calculate normalization constant
            for nei_point in graph_points[neighbor_index]:
                W = W + 1.0 / math.sqrt( (nei_point[0] - point[0])**2 + (nei_point[1] - point[1])**2 + (nei_point[2] - point[2])**2 )
            # calculate Laplace coordinates
            for nei_point in graph_points[neighbor_index]:
                mesh1.append(point)
                mesh2.append(nei_point)
                dis_nei = math.sqrt( (nei_point[0] - point[0])**2 + (nei_point[1] - point[1])**2 + (nei_point[2] - point[2])**2 )
                Lap = Lap - nei_point / ( dis_nei * W )
            InterMesh[idx] = Lap
        state5 = InterMesh.flatten()

        # DisMesh = np.empty([limb_pose.shape[0], self.target_line.shape[0]])
        # for i in range(DisMesh.shape[0]):
        #     for j in range(DisMesh.shape[1]):
        #         DisMesh[i][j] = math.sqrt( (limb_pose[i][0] - self.target_line[j][0]) ** 2 +
        #                                    (limb_pose[i][1] - self.target_line[j][1]) ** 2 +
        #                                    (limb_pose[i][2] - self.target_line[j][2]) ** 2)
        # state5 = DisMesh.flatten()
        #
        # DisMesh = np.empty(limb_pose.shape)
        # for i in range(DisMesh.shape[0]):
        #     W = 0
        #     Lap = limb_pose[i]
        #     # calculate normalization constant
        #     for target_point in self.target_line:
        #         W = W + 1.0 / math.sqrt((limb_pose[i][0] - target_point[0])**2 +
        #                                 (limb_pose[i][1] - target_point[1])**2 +
        #                                 (limb_pose[i][2] - target_point[2])**2)
        #     for target_point in self.target_line:
        #         dis = math.sqrt((limb_pose[i][0] - target_point[0])**2 +
        #                         (limb_pose[i][1] - target_point[1])**2 +
        #                         (limb_pose[i][2] - target_point[2])**2)
        #         Lap = Lap - target_point / (dis * W)
        #     DisMesh[i] = Lap
        # state5 = DisMesh.flatten()

        state = [state1, state2, state3, state4, state5]
        return state, writhe, InterMesh, [mesh1, mesh2]

    def act(self, action):

        # ########### Joint torque control ####################
        # limb_interface.set_joint_torques(cmd)

        action_right = action[0:7]
        action_left= action[7:14]
        # right arm
        t1 = threading.Thread(target=movearm_job, args=(self.right_limb_interface, action_right))
        t1.start()
        # left arm
        t2 = threading.Thread(target=movearm_job, args=(self.left_limb_interface, action_left))
        t2.start()
        t2.join()
        t1.join()


    def load_model(self, name, path, block_pose,
                    block_reference_frame="world", type="sdf"):
        # Get Models' Path
        model_path = "./gazebo_models/"

        # Load Block SDF
        block_xml = ''
        with open(model_path + path, "r") as block_file:
            block_xml = block_file.read().replace('\n', '')

        # Spawn Block SDF
        resp = 0
        if type == "sdf":
            rospy.wait_for_service('/gazebo/spawn_sdf_model')
            try:
                spawn_sdf = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
                resp = spawn_sdf(name, block_xml, "/",
                                       block_pose, block_reference_frame)
            except rospy.ServiceException, e:
                rospy.logerr("Spawn SDF service call failed: {0}".format(e))

        if type == "urdf":
            rospy.wait_for_service('/gazebo/spawn_urdf_model')
            try:
                spawn_urdf = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)
                resp = spawn_urdf(name, block_xml, "/",
                                       block_pose, block_reference_frame)
            except rospy.ServiceException, e:
                rospy.logerr("Spawn URDF service call failed: {0}".format(e))

        return resp

    def delete_model(self, name):
        try:
            delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
            resp_delete = delete_model(name)
        except rospy.ServiceException, e:
            rospy.loginfo("Delete Model service call failed: {0}".format(e))

    def clear(self):
        self.delete_model("humanoid")
        rospy.sleep(0.1)


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
        print "Joint angles error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
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
            limb_joint[i][j] = kdl_array[j]


    for i in range(arm_chain.getNrOfSegments()):
        joint_array = limb_joint[limb_chain[i].getNrOfJoints()-1]
        limb_fk[i].JntToCart(joint_array,  limb_frame[i])
        pos = limb_frame[i].p
        rot = PyKDL.Rotation(limb_frame[i].M)
        rot = rot.GetQuaternion()
        limb_pose.append( [pos[0], pos[1], pos[2]] )

    return np.asarray(limb_pose), kdl_array


def movearm_job(limb_interface, action):
    cmd = dict()
    # ########## delta Joint position control ###############
    limb_interface.set_joint_position_speed(0.1)
    for i, joint in enumerate(limb_interface.joint_names()):
        cmd[joint] = action[i]
    cur_type_values = limb_interface.joint_angles()
    for i, joint in enumerate(limb_interface.joint_names()):
        cmd[joint] = cmd[joint] + cur_type_values[joint]
    try:
        limb_interface.move_to_joint_positions(cmd, timeout=2.0)
        # limb_interface.set_joint_positions(cmd, raw=False)
    except Exception, e:
        rospy.logerr('Error: %s', str(e))

    # ########## Joint velocity control ############
    # for i, joint in enumerate(limb_interface.joint_names()):
    #     cmd[joint] = action[i]
    # try:
    #     limb_interface.set_joint_velocities(cmd)
    # except Exception, e:
    #     rospy.logerr('Error: %s', str(e))
    # rospy.sleep(1)

def resetarm_job(limb_interface, start_point):
    # right arm
    cmd = dict()
    for i, joint in enumerate(limb_interface.joint_names()):
        cmd[joint] = start_point[i]
    try:
        limb_interface.move_to_joint_positions(cmd, timeout=5.0)
    except Exception, e:
        rospy.logerr('Error: %s', str(e))









