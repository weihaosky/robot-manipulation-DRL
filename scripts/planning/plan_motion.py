from planning_scene import IKsolver, RobotMove

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

        lower_limits = [-1.7016, -2.147, -3.0541, -0.05, -3.059, -1.5707, -3.509]
        upper_limits = [1.7016, 1.047, 3.0541, 2.618, 3.059, 2.094, 3.509]

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
    def __init__(self, space_information, iksolver, ndof):
        super(MyStateValidityChecker, self).__init__(space_information)
        self.space_information = space_information
        self.iksolver = iksolver
        self.ndof = ndof

    def isValid(self, state):
        valid_bound = self.space_information.satisfiesBounds(state)
        joint_angles = range(self.ndof)
        for i in range(self.ndof):
            joint_angles[i] = state[i]
        valid_collision = self.iksolver.collision_check(joint_angles).valid
        return (valid_bound and valid_collision)


class MyOptimizationObjective(ob.PathLengthOptimizationObjective):
    def __init__(self, si):
        super(MyOptimizationObjective, self).__init__(si)
        self.setCostThreshold(ob.Cost(10))


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

        # w = self.iksolver.forward(state_vector)
        # dis = 0.9 - w

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

        vc = MyStateValidityChecker(self.simple_setup.getSpaceInformation(), self.iksolver, ndof)
        self.simple_setup.setStateValidityChecker(vc)

        ob = MyOptimizationObjective(self.simple_setup.getSpaceInformation())
        self.simple_setup.setOptimizationObjective(ob)

        propagator = MyStatePropagator(self.simple_setup.getSpaceInformation(), ndof)
        self.simple_setup.setStatePropagator(propagator)

        # self.planner = oc.KPIECE1(self.simple_setup.getSpaceInformation())
        # self.planner.setup()
        # ========= RRT planner ============
        self.planner = og.RRTstar(self.simple_setup.getSpaceInformation())
        p_goal = 0.0
        self.planner.setGoalBias(p_goal)
        self.planner.setRange(step_size)

        IPython.embed()
        # =========== PRM planner =============
        # self.planner = og.PRM(self.simple_setup.getSpaceInformation())

        self.simple_setup.setPlanner(self.planner)

    def solve(self, start, goal, timeout):
        self.simple_setup.setStartState(start)
        # mygoal = MyGoal(self.simple_setup.getSpaceInformation(), goal, self.ndof)
        mygoalregion = MyGoalRegion(self.simple_setup.getSpaceInformation(), goal, self.iksolver, self.ndof)
        self.simple_setup.setGoal(mygoalregion)
        # self.simple_setup.setGoalState(goal)
        self.simple_setup.setup()

        if self.simple_setup.solve(timeout):
            if self.simple_setup.haveExactSolutionPath():
                print ("Exact Solution.")
                return self.simple_setup.getSolutionPath()
            elif self.simple_setup.haveSolutionPath():
                print ("Approximate Solution.")
                return self.simple_setup.getSolutionPath()
        else:
            print ("No Solution Found.")
            return None


if __name__ == '__main__':

    rospy.init_node('baxter_planning')

    ndof = 7
    limb = "right"

    robot_move = RobotMove()

    iksolver = IKsolver(limb)
    iksolver.reset()
    iksolver.update_human()

    planner = MyRRT(ndof, iksolver, 0.1)

    # get start joint state
    joint_angles = iksolver._right_limb_interface.joint_angles()
    joint_names = iksolver._right_limb_interface.joint_names()
    start_vector = np.empty(ndof)
    for idx, name in enumerate(joint_names):
        start_vector[idx] = joint_angles[name]
    start = ob.State(planner.state_space)
    goal = ob.State(planner.state_space)

    # start_vector = np.array([0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    # goal_vector = np.array([-0.3, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for i in range(ndof):
        start[i] = start_vector[i]
        # goal[i] = goal_vector[i]

    # ============== plan ===============
    result_path = planner.solve(start, goal, 200)
    path = []
    joints = np.empty(ndof)
    for i in range(result_path.getStateCount()):
        for j in range(ndof):
            joints[j] = result_path.getStates()[i][j]
        path.append(copy.deepcopy(joints))

    IPython.embed()
    if result_path != None:
        robot_move.execute(path)







