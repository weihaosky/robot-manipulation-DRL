from acnetwork import *
from Env_control import *

import sys
import copy
import rospy
import rospkg
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import tf as transfor
import math
import random
import numpy as np
from moveit_msgs.msg import Constraints, OrientationConstraint
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from gazebo_msgs.srv import SpawnModel, DeleteModel, GetModelState
from sensor_msgs.msg import Image

import os
import threading
import IPython
import scipy.signal
from collections import deque
import pickle

import numpy as np
import time
import matplotlib.pyplot as plt

import cv2
from cv_bridge import CvBridge, CvBridgeError



def preprocess(observation):
    observation = observation[250:600, 200:550, ::]
    x = cv2.resize(observation, (Rows, Cols))
    x[x == 155] = 235
    x[x == 129] = 235
    gauss = np.random.normal(0, 0, x.shape)
    x_noisy = x + gauss
    cv2.imwrite('Image/x%d.jpeg'%Frame, x_noisy)
    x_noisy = x_noisy.reshape(1, Rows, Cols, Channels)
    x_noisy = x_noisy.astype(np.uint8)
    return x_noisy

def hook_visu(self, input, output):
    print output.data.size()
    feamap = output.data.cpu().numpy()
    plt.figure(1)
    plt.title("Image used")
    plt.imshow(x[0], interpolation="nearest")
    plt.figure(2)
    plt.title("Convolution:")
    n_columns = 12
    n_rows = 9  # math.ceil(features.shape[3] / n_columns) + 1
    for i in range(0, 108):  # features.shape[3]):
        plt.subplot(n_rows, n_columns, 1 + i)
        plt.imshow(feamap[0, i, :, :], interpolation="nearest", cmap="gray")
        plt.axis('off')
    plt.show()

def visualize(x, net):
    x = Variable(torch.from_numpy(x).cuda(), volatile=True)
    net.max4.register_forward_hook(hook_visu)
    value = net(x)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.xavier_normal(m.weight.data, 1)
        torch.nn.init.constant(m.bias.data, 0.0)
    # elif classname.find('Linear') != -1:
    #     torch.nn.init.xavier_uniform(m.weight.data, 1)
    #     torch.nn.init.constant(m.bias.data, 0.0)


print "============ Starting setup ============"
moveit_commander.roscpp_initialize(sys.argv)
rospy.init_node('move_group_python_interface_tutorial',
                anonymous=True)

robot = moveit_commander.RobotCommander()
scene = moveit_commander.PlanningSceneInterface()
group = moveit_commander.MoveGroupCommander("left_arm")


image_getter = InfoGetter()
image_topic = "/camera1/image_raw"
rospy.Subscriber(image_topic, Image, image_getter)


C1 = Collision_object("C1")
C2 = Collision_object("C2")

# move robot to the starting position
try:
    env_reset(group, load_block=False)
except Exception, e:
    rospy.logerr('Error on env_reset: %s', str(e))

load_table_models()
# Remove models from the scene on shutdown
rospy.on_shutdown(delete_gazebo_models)

# load block object, collision obstacle
try:
    env_reset(group)
except Exception, e:
    rospy.logerr('Error on move_baxter: %s', str(e))


##############################reinforcement learning####################################
model_path = "./model_baxter_net/"
if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists("./Image"):
    os.makedirs("./Image")

np.set_printoptions(threshold=np.nan)

Rows = 128
Cols = 128
Channels = 3
# learning_rate = 1e-4
gamma = 0.99 # discouting factor
TAU = 0.001 # target-net learning rate
OBSERVE = 5 # donot update network
COUNT = 0

running_reward = None
reward_sum = 0
episode_number = 1
xs, acs, rs= [], [], []
vs = []
Frame = 0   # constrain the max frames to prevent memory over use
Block_position_last = []     # for the calculation of reward1 in one frame
Distance = 0    # for the calculation of reward2 in one frame
Object_last = 0  # for the calculation of reward1 in one frame
Record = []
Record_mean = []
Update_count = 0
Step_count = 0

# capture image
try:
    observation = capture_environment(image_getter, Frame)
except Exception, e:
    rospy.logerr('Error on capture_env: %s', str(e))

suc_rate_buffer = deque()       # for the calculation of success rate

if 1:
    replayBuffer = ReplayBuffer()
else:
    file_load = open('Buffer/Buffer_collision-fail.pkl', 'rb')
    replayBuffer = pickle.load(file_load)

primary_net = Network([Channels, Rows, Cols], 5, "primary")
target_net = Network([Channels, Rows, Cols], 5, "target")
# primary_net.apply(weights_init)
# target_net.load_state_dict(primary_net.state_dict())

use_cuda = torch.cuda.is_available()
if use_cuda:
    primary_net = primary_net.cuda()
    target_net = target_net.cuda()


resume = False  # whether load previous model
if resume:
    print ('Loading Model...')
    primary_net.load_state_dict(torch.load(model_path + "model_p-5000.pt"))
    target_net.load_state_dict(torch.load(model_path + "model_t-5000.pt"))
    episode_number = 5000
    OBSERVE = 5100
else:
    print ("Initializing Model...")


start_time = time.time()
episode_rewards = []  # for summary
episode_mean_values = []  # for summary
episode_collision_move = [] # for summary

while not rospy.is_shutdown():
    # rospy.logerr("WE ARE HERE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    try:
        observation = capture_environment(image_getter, Frame)
    except Exception, e:
        rospy.logerr('Error on capture_env: %s', str(e))
        IPython.embed()
    x = preprocess(observation)

    if Frame == 0:
        reward, done, success, COLLISION, Frame, Block_position_last, Distance, Object_last = state_evaluate(group, Frame,
                                                                                                         Block_position_last,
                                                                                                         Distance,
                                                                                                         Object_last)
    # ################################ visualization ###########################################
    # visualize(x, target_net)
    # visualize(x, primary_net)
    # ##########################################################################################

    action, actions, aprob, value = select_action(x, target_net, episode_number, Distance, Block_position_last, Target_x)

    print "value is %f " % value
    print "actions is ",
    print actions,
    print "action is %d" % action

    try:
        move_baxter(group, action)
        Step_count+=1
    except Exception, e:
        rospy.logerr('Error on move_baxter: %s', str(e))
        IPython.embed()

    try:
        reward, done, success, COLLISION, Frame, Block_position_last, Distance, Object_last = state_evaluate(group, Frame, Block_position_last, Distance, Object_last)
    except Exception, e:
        rospy.logerr('Error on state_evaluate: %s', str(e))
        IPython.embed()

    xs.append(x)  # all observations in a episode
    acs.append(action)  # all action prob labels in a episode
    rs.append(reward)  # all rewards in a episode
    vs.append(value)

    if done:
        episode_number += 1
        try:
           observation = capture_environment(image_getter, Frame)
        except Exception, e:
            rospy.logerr('Error on capture_env: %s', str(e))
            IPython.embed()
        x1 = preprocess(observation)

        if success == 1:
            rs_add = np.zeros_like(rs)
            rs_add[-1] = 1.0
            rs_add = np.asarray(rs_add)
            rs_add = discount(rs_add, gamma)
            rs = np.asarray(rs)
            rs = rs + rs_add
            rs = rs.tolist()
        else:
            rs_add = np.zeros_like(rs)
            rs_add[-1] = -1.0
            rs_add = np.asarray(rs_add)
            if COLLISION == 0:
                rs_add = discount(rs_add, gamma)
            else:                   # if collision, the main fault is the last few movements
                rs_add = discount(rs_add, 0.95)
            rs = np.asarray(rs)
            rs = rs + rs_add
            rs = rs.tolist()

        Frame = 0

        x1s = xs[1:]
        x1s.append(x1)

        episode_mean_values.append(np.mean(vs)) #for summary

        # epx = np.vstack(xs)
        # epy = np.asarray(ys)
        epr = np.asarray(rs)
        # epv = np.asarray(vs)
        print "Discounted Reward is:"
        print epr

        if episode_number <= OBSERVE:
            if epr.shape[0] >= 5:
                for i in range(epr.shape[0]):
                    replayBuffer.setPerception(np.squeeze(xs[i]), acs[i], rs[i], np.squeeze(x1s[i]), success)
        else:
            success_batch = [data[4] for data in replayBuffer.replayMemory]
            success_ratio = float(success_batch.count(1))/float(len(success_batch))
            print "buffer_length:", len(success_batch)
            print "success ratio:", success_ratio

            # ############################ Buffer updating ##############################
            if epr.shape[0] >= 5:  # not the bug episode
                # if success_ratio < 0.3:  # mainly store success experience
                #     if success == 1:
                #         for i in range(epr.shape[0]):
                #             replayBuffer.setPerception(np.squeeze(xs[i]), acs[i], rs[i], np.squeeze(x1s[i]), success)
                #     else:
                #         if random.uniform(0, 1) < 0.05:
                #             for i in range(epr.shape[0]):
                #                 replayBuffer.setPerception(np.squeeze(xs[i]), acs[i], rs[i], np.squeeze(x1s[i]),
                #                                            success)
                # if 0.3 <= success_ratio:  # store all experience
                #     for i in range(epr.shape[0]):
                #         replayBuffer.setPerception(np.squeeze(xs[i]), acs[i], rs[i], np.squeeze(x1s[i]), success)
                for i in range(epr.shape[0]):
                    replayBuffer.setPerception(np.squeeze(xs[i]), acs[i], rs[i], np.squeeze(x1s[i]), success)

            # ########################## training #################################
            # if 0.3 <= success_ratio <= 0.7:  # conduct the training
            #     print "updating network..."
            #     for i in range(6):
            #         loss = train(target_net, primary_net, replayBuffer)
            #         updateTarget(target_net,
            #                      primary_net)  # Update the target network toward the primary network.
            #         Update_count += 1
            # else:
            #     if 0.2 <= success_ratio <= 0.8:  # conduct the training
            #         print "updating network..."
            #         for i in range(4):
            #             loss = train(target_net, primary_net, replayBuffer)
            #             updateTarget(target_net,
            #                          primary_net)  # Update the target network toward the primary network.
            #             Update_count += 1
            #     else:
            #         if 0.1 <= success_ratio:
            #             print "updating network..."
            #             for i in range(2):
            #                 loss = train(target_net, primary_net, replayBuffer)
            #                 updateTarget(target_net,
            #                              primary_net)  # Update the target network toward the primary network.
            #                 Update_count += 1
            #         else:
            #             if random.uniform(0, 1) < 0.3:  # small probability conduct training
            #                 print "updating network...small probability"
            #                 loss = train(target_net, primary_net, replayBuffer)
            #                 updateTarget(target_net,
            #                              primary_net)  # Update the target network toward the primary network.
            #                 Update_count += 1
            print "updating network..."
            for i in range(2):
                loss = train(target_net, primary_net, replayBuffer)
                updateTarget(target_net, primary_net)  # Update the target network toward the primary network.
                Update_count += 1
            # #####################################################################

        # ###################### calculate the success rate ###############################
        suc_rate_buffer.append(success)
        if len(suc_rate_buffer) > 100:
            suc_rate_buffer.popleft()
        suc_rate100 = float(suc_rate_buffer.count(1)) / float(len(suc_rate_buffer))
        print "suc_buffer_length:",
        print len(suc_rate_buffer),
        print "success rate for the last 100 episodes:",
        print  suc_rate100
        # ###################### calculate the success rate ###############################

        reward_sum = np.mean(epr)
        xs, acs, rs = [], [], []
        vs = []

        if episode_number > OBSERVE:
            record = [suc_rate100, reward_sum, success_ratio, Update_count, Step_count]
            print "record:",
            print record
            Record.append(record)
            if episode_number % 50 == 0:
                file_save = open('Record.pkl', 'wb')
                pickle.dump(Record, file_save)
                file_save.close()

        episode_rewards.append(reward_sum)  #for summary
        # reward_sum = 0
        if episode_number % 5 == 0 and episode_number > OBSERVE:
            mean_reward = np.mean(episode_rewards[-5:])
            mean_value = np.mean(episode_mean_values[-5:])

        if episode_number % 250 == 0:
            print "saving model..."
            torch.save(primary_net.state_dict(), model_path + 'model_p-' + str(episode_number) + '.pt')
            torch.save(target_net.state_dict(), model_path + 'model_t-' + str(episode_number) + '.pt')

        # if episode_number % 1000 == 0:
        #     file_save = open('/media/will/D/Ubuntu/Buffer.pkl', 'wb')
        #     pickle.dump(replayBuffer, file_save)
        #     file_save.close()

        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        print('resetting env. episode reward mean was %f. running mean: %f' % (reward_sum, running_reward))

        delete_models("block")
        delete_models("target")
        C1.remove_collision_object()
        C2.remove_collision_object()
        try:
            env_reset(group, move_up=True)  # reset env
        except Exception, e:
            rospy.logerr('Error on move_baxter: %s', str(e))
        rospy.sleep(0.5)

    print(('**************** episode %d: move %d took %.5fs, reward: %f ***************' %
                (episode_number, Frame, time.time()-start_time, reward)))
    start_time = time.time()


moveit_commander.roscpp_shutdown()