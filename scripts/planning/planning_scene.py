
import numpy as np
import math
import sys, os, inspect
import copy
import rospy

# =========== iksolver + collision checking
import baxter_interface
import PyKDL
from baxter_kdl.kdl_kinematics import KDLKinematics
from baxter_kdl.kdl_parser import kdl_tree_from_urdf_model
from urdf_parser_py.urdf import URDF
from tf import transformations
from gazebo_msgs.srv import GetModelState, GetLinkState, SetModelConfiguration
import moveit_commander
from moveit_msgs.srv import GetStateValidityRequest, GetStateValidity
from moveit_msgs.msg import RobotState
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose
import moveit_python

# ========= movement
import actionlib
import threading
from control_msgs.msg import (
    FollowJointTrajectoryAction,
    FollowJointTrajectoryGoal,
)
from trajectory_msgs.msg import (
    JointTrajectoryPoint,
)

# ======= tools
import IPython
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from utils import GLI


class IKsolver():
    def __init__(self, limb):
        self._baxter = URDF.from_parameter_server(key='robot_description')
        self._kdl_tree = kdl_tree_from_urdf_model(self._baxter)
        self._right_limb_interface = baxter_interface.Limb('right')
        self._base_link = self._baxter.get_root()
        self._tip_link = limb + '_gripper'
        self.solver = KDLKinematics(self._baxter, self._base_link, self._tip_link)
        self.rs = RobotState()
        self.rs.joint_state.name = ['right_s0', 'right_s1', 'right_e0', 'right_e1',
                                    'right_w0', 'right_w1', 'right_w2']
        self.rs.joint_state.position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # Moveit group setup
        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_python.PlanningSceneInterface(self.robot.get_planning_frame())
        # self.scene = moveit_commander.PlanningSceneInterface()
        self.group_name = "right_arm"
        self.group = moveit_commander.MoveGroupCommander(self.group_name)

        # service
        self.set_model_config = rospy.ServiceProxy('/gazebo/set_model_configuration', SetModelConfiguration)
        self.get_link_state = rospy.ServiceProxy("/gazebo/get_link_state", GetLinkState)
        self.sv_srv = rospy.ServiceProxy('/check_state_validity', GetStateValidity)

        # human target
        self.target_pos_start = np.asarray([0.5, 0, -0.93])  # robot /base frame, z = -0.93 w.r.t /world frame
        self.target_line_start = np.empty([22, 3], float)
        for i in range(11):
            self.target_line_start[i] = self.target_pos_start + [0, -0.0, 1.8] - (
                        np.asarray([0, -0.0, 1.8]) - np.asarray([0, -0.0, 0.5])) / 10 * i
            self.target_line_start[i + 11] = self.target_pos_start + [0, -0.5, 1.3] + (
                        np.asarray([0, 0.5, 1.3]) - np.asarray([0, -0.5, 1.3])) / 10 * i
        self.target_line = self.target_line_start

    def reset(self):
        # ================reset baxter joint position=================
        # print("Resetting Baxter...")
        # cmd = dict()
        # start_point_right = [-0.3, 1.0, 0.0, 0.5, 0.0, 0.027, 0.0]
        # for i, joint in enumerate(self._right_limb_interface.joint_names()):
        #     cmd[joint] = start_point_right[i]
        # self._right_limb_interface.set_joint_positions(cmd, raw=False)
        # self.set_model_config('baxter', 'robot_description',
        #                       ['right_s0', 'right_s1', 'right_e0', 'right_e1', 'right_w0', 'right_w1', 'right_w2'],
        #                       [-0.3, 1.0, 0.0, 0.5, 0.0, 0.027, 0.0])
        # Moveit joint move
        joint_goal = self.group.get_current_joint_values()
        # s0, s1, e0, e1, w0, w1, w2
        joint_goal[0] = -0.3
        joint_goal[1] = 1.0
        joint_goal[2] = 0.0
        joint_goal[3] = 0.5
        joint_goal[4] = 0.0
        joint_goal[5] = 0.027
        joint_goal[6] = 0.0
        self.group.go(joint_goal, wait=True)
        self.group.stop()

        # ============add human target collision=============
        if 'target' not in self.scene.getKnownCollisionObjects():
            humanoid_pose = Pose()
            humanoid_pose.position.x = 0.6
            humanoid_pose.position.y = 0
            humanoid_pose.position.z = -0.93
            self.scene.addMesh("target", humanoid_pose, "gazebo_models/humanoid/stl/humanoid-fat.stl")
            rospy.sleep(0.1)
            print "adding target..."

    def update_human(self):
        rospy.wait_for_service('/gazebo/get_link_state')
        torso_pose = self.get_link_state("humanoid::Torso_link", "world").link_state.pose
        T = transformations.quaternion_matrix([torso_pose.orientation.x,
                                               torso_pose.orientation.y,
                                               torso_pose.orientation.z,
                                               torso_pose.orientation.w])
        # rotation in /world frame, which is [0, 0, -0.93] to /base frame
        # target_line_start-target_pos_start is the original position of human in /world frame
        self.target_line = np.dot(T[:3, :3], (self.target_line_start - self.target_pos_start).T).T + \
                           [torso_pose.position.x, torso_pose.position.y, torso_pose.position.z] + \
                           [0, 0, -0.93]

    def forward(self, joint_angles):
        # Calculate writhe
        right_limb_pose, _ = limbPose(self._kdl_tree, self._base_link, self._right_limb_interface, joint_angles)
        # left_limb_pose, _ = limbPose(self._kdl_tree, self._base_link, self._left_limb_interface, joint_angles, 'left')
        writhe = np.empty((len(self.target_line) - 2, 14))
        for idx_target in range(10):
            for idx_robot in range(5, 12):
                x1_right = self.target_line[idx_target].copy()
                x2_right = self.target_line[idx_target + 1].copy()
                x1_right[1] -= 0.15
                x2_right[1] -= 0.15
                writhe[idx_target, idx_robot - 5] = GLI(x1_right, x2_right,
                                                        right_limb_pose[idx_robot], right_limb_pose[idx_robot + 1])[0]
                # x1_left = self.target_line[idx_target].copy()
                # x2_left = self.target_line[idx_target + 1].copy()
                # x1_left[1] += 0.15
                # x2_left[1] += 0.15
                # writhe[idx_target, idx_robot - 5 + 7] = GLI(x1_left, x2_left,
                #                                             left_limb_pose[idx_robot], left_limb_pose[idx_robot + 1])[0]

        for idx_target in range(11, 21):
            for idx_robot in range(5, 12):
                writhe[idx_target - 1, idx_robot - 5] = \
                    GLI(self.target_line[idx_target], self.target_line[idx_target + 1],
                        right_limb_pose[idx_robot], right_limb_pose[idx_robot + 1])[0]
                # writhe[idx_target - 1, idx_robot - 5 + 7] = \
                #     GLI(self.target_line[idx_target], self.target_line[idx_target + 1],
                #         left_limb_pose[idx_robot], left_limb_pose[idx_robot + 1])[0]
        w_right1 = np.abs(writhe[0:10, 0:7].flatten().sum())
        w_right2 = np.abs(writhe[10:20, 0:7].flatten().sum())
        # w_left1 = np.abs(writhe[0:10, 7:14].flatten().sum())
        # w_left2 = np.abs(writhe[10:20, 7:14].flatten().sum())
        w = w_right1 + w_right2

        return w

    def collision_check(self, joint_angles):
        gsvr = GetStateValidityRequest()
        self.rs.joint_state.position = joint_angles
        gsvr.robot_state = self.rs
        gsvr.group_name = self.group_name

        result = self.sv_srv.call(gsvr)
        return result




def limbPose(kdl_tree, base_link, limb_interface, joint_angles, limb = 'right'):
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
    kdl_array = PyKDL.JntArray(num_jnts)
    for idx in range(len(joint_angles)):
        kdl_array[idx] = joint_angles[idx]

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


class RobotMove():
    def __init__(self):
        # create our action server clients
        self._left_client = actionlib.SimpleActionClient(
            'robot/limb/left/follow_joint_trajectory',
            FollowJointTrajectoryAction,
        )
        self._right_client = actionlib.SimpleActionClient(
            'robot/limb/right/follow_joint_trajectory',
            FollowJointTrajectoryAction,
        )
        self.set_model_config = rospy.ServiceProxy('/gazebo/set_model_configuration', SetModelConfiguration)

        # verify joint trajectory action servers are available
        l_server_up = self._left_client.wait_for_server(rospy.Duration(10.0))
        r_server_up = self._right_client.wait_for_server(rospy.Duration(10.0))
        if not l_server_up or not r_server_up:
            msg = ("Action server not available."
                   " Verify action server availability.")
            rospy.logerr(msg)
            rospy.signal_shutdown(msg)
            sys.exit(1)
        # create our goal request
        self._l_goal = FollowJointTrajectoryGoal()
        self._r_goal = FollowJointTrajectoryGoal()

        # limb interface - current angles needed for start move
        self._l_arm = baxter_interface.Limb('left')
        self._r_arm = baxter_interface.Limb('right')


    def reset(self):
        start_point_right = [-0.3, 1.0, 0.0, 0.5, 0.0, 0.027, 0.0]
        self.execute([np.array(start_point_right)])
        # self.set_model_config('baxter', 'robot_description',
        #                       ['right_s0', 'right_s1', 'right_e0', 'right_e1', 'right_w0', 'right_w1', 'right_w2'],
        #                       [-0.3, 1.0, 0.0, 0.5, 0.0, 0.027, 0.0])

    def add_point(self, positions, goal, time):
        # creates a point in trajectory with time_from_start and positions
        point = JointTrajectoryPoint()
        point.positions = copy.copy(positions)
        point.time_from_start = rospy.Duration(time)
        goal.trajectory.points.append(point)


    def execute(self, path):
        self._l_goal = FollowJointTrajectoryGoal()
        self._r_goal = FollowJointTrajectoryGoal()

        self._r_goal.trajectory.joint_names = self._r_arm.joint_names()
        self._l_goal.trajectory.joint_names = self._l_arm.joint_names()

        for i in range(len(path)):
            self.add_point(path[i].tolist(), self._r_goal, 0.1*i)

        """
        Sends FollowJointTrajectoryAction request
        """
        self._right_client.send_goal(self._r_goal)

        self.wait()

    def wait(self, timeout=15.0):
        self._right_client.wait_for_result(timeout=rospy.Duration(timeout))
