from iksolver import IKsolver

import numpy as np
import math
import sys
import copy
import rospy



# ======= ompl
from ompl import base as ob
from ompl import control as oc
from ompl import geometric as og

# ======= tools
import IPython


class MyStateSpace(ob.RealVectorStateSpace):
    def __init__(self, ndof):
        self.ndof = ndof
        super(MyStateSpace, self).__init__(self.ndof)

        lower_limits = [-3.059, -1.57079632679, -3.059, -0.05, -3.05417993878, -2.147, -1.70167993878]
        upper_limits = [3.059, 2.094, 3.059, 2.618, 3.05417993878, 1.047, 1.70167993878]

        joint_bounds = ob.RealVectorBounds(self.ndof)
        for i in range(self.ndof):
            joint_bounds.setLow(i, lower_limits[i])
            joint_bounds.setHigh(i, upper_limits[i])

        self.setBounds(joint_bounds)
        self.setup()


class MyControlSpace(oc.RealVectorControlSpace):
    def __init__(self, state_space, ndof):
        self.ndof = ndof
        super(MyControlSpace, self).__init__(state_space, self.ndof)

        joint_velocity_limit_bounds = ob.RealVectorBounds(self.ndof)
        joint_velocities = [3.15, 3.15, 3.15, 3.2, 3.2, 3.2, 3.2]

        for i in range(self.ndof):
            joint_velocity_limit_bounds.setLow(i, -joint_velocities[i])
            joint_velocity_limit_bounds.setHigh(i, joint_velocities[i])

        self.setBounds(joint_velocity_limit_bounds)


class MyStateValidityChecker(ob.StateValidityChecker):
    def __init__(self, space_information):
        super(MyStateValidityChecker, self).__init__(space_information)
        self.space_information = space_information

    def isValid(self, state):
        return self.space_information.satisfiesBounds(state)


class MyGoal(ob.Goal):
    def __init__(self, space_information, goal_config, ndof):
        super(MyGoal, self).__init__(space_information)
        self.goal_config = goal_config
        self.ndof = ndof

    def isSatisfied(self, state):
        start_state_vector = [state[i] for i in range(self.ndof)]
        goal_state_vector = [self.goal_config[i] for i in range(self.ndof)]
        joints_are_within_threshold = [abs(a - b) < 0.15 for a, b in zip(goal_state_vector, start_state_vector)]
        return all(joints_are_within_threshold)


class MyGoalRegion(ob.GoalRegion):
    def __init__(self, si, goal_config, iksolver, ndof):
        super(MyGoalRegion, self).__init__(si)
        self.setThreshold(0.1)
        self.goal_config = goal_config
        self.iksolver = iksolver
        self.ndof = ndof

    def distanceGoal(self, state):
        state_vector = [state[i] for i in range(self.ndof)]
        goal_state_vector = [self.goal_config[i] for i in range(self.ndof)]
        joints_diff = [abs(a - b) for a, b in zip(goal_state_vector, state_vector)]
        dis = np.asarray(joints_diff).sum()

        w = self.iksolver.forward(state_vector)
        dis = 0.8 - w

        return dis


class MyStatePropagator(oc.StatePropagator):
    def __init__(self, space_information, ndof):
        super(MyStatePropagator, self).__init__(space_information)
        self.ndof = ndof
        self.space_information = space_information

    def propagate(self, state, control, duration, result):
        for i in range(self.ndof):
            joint_value = state[i]
            joint_velocity = control[i]
            result[i] = joint_value + joint_velocity * duration


def directedControlSamplerAllocator(si):
    print "Here"
    sampler = oc.SimpleDirectedControlSampler(si, 10)
    return sampler


class MyRRT:
    def __init__(self, ndof, iksolver, step_size=0.05):
        self.ndof = ndof
        self.iksolver = iksolver
        self.state_space = MyStateSpace(ndof)
        self.control_space = MyControlSpace(self.state_space, ndof)
        self.simple_setup = oc.SimpleSetup(self.control_space)
        si = self.simple_setup.getSpaceInformation()
        si.setPropagationStepSize(step_size)
        si.setMinMaxControlDuration(1, 1)
        si.setDirectedControlSamplerAllocator(oc.DirectedControlSamplerAllocator(directedControlSamplerAllocator))

        vc = MyStateValidityChecker(self.simple_setup.getSpaceInformation())
        self.simple_setup.setStateValidityChecker(vc)

        propagator = MyStatePropagator(self.simple_setup.getSpaceInformation(), ndof)
        self.simple_setup.setStatePropagator(propagator)

        # self.planner = oc.KPIECE1(self.simple_setup.getSpaceInformation())
        # self.planner.setup()
        self.planner = og.RRTstar(self.simple_setup.getSpaceInformation())
        p_goal = 0.0
        self.planner.setGoalBias(p_goal)
        self.planner.setRange(step_size)

        self.simple_setup.setPlanner(self.planner)

    def solve(self, start, goal):
        self.simple_setup.setStartState(start)
        # mygoal = MyGoal(self.simple_setup.getSpaceInformation(), goal, self.ndof)
        mygoalregion = MyGoalRegion(self.simple_setup.getSpaceInformation(), goal, self.iksolver, self.ndof)
        self.simple_setup.setGoal(mygoalregion)
        # self.simple_setup.setGoalState(goal)
        self.simple_setup.setup()

        if self.simple_setup.solve(10.0):
            if self.simple_setup.haveExactSolutionPath():
                print ("Exact Solution.")
                return self.simple_setup.getSolutionPath().printAsMatrix()
            elif self.simple_setup.haveSolutionPath():
                print ("Approximate Solution.")
                return self.simple_setup.getSolutionPath().printAsMatrix()
        else:
            print ("No Solution Found.")
            return None


if __name__ == '__main__':

    rospy.init_node('baxter_hug')
    #
    # moveit_commander.roscpp_initialize(sys.argv)
    # robot = moveit_commander.RobotCommander()
    # scene = moveit_python.PlanningSceneInterface(robot.get_planning_frame())
    # # self.scene = moveit_commander.PlanningSceneInterface()
    # right_group_name = "right_arm"
    # right_group = moveit_commander.MoveGroupCommander(right_group_name)

    # start_pose = right_group.get_current_pose().pose
    # goal_pose = copy.deepcopy(start_pose)
    # goal_pose.position.x = start_pose.position.x + 0.1

    # rrt = RRT("right")
    #
    # rrt.plan(pose_to_7x1_vector(start_pose),
    #          pose_to_7x1_vector(goal_pose))

    ndof = 7
    limb = "right"

    iksolver = IKsolver(limb)
    planner = MyRRT(ndof, iksolver)
    start = ob.State(planner.state_space)
    goal = ob.State(planner.state_space)

    # Create a very simple problem.
    start_vector = np.array([0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    goal_vector = np.array([-0.3, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for i in range(ndof):
        start[i] = start_vector[i]
        goal[i] = goal_vector[i]

    result = planner.solve(start, goal)

    IPython.embed()