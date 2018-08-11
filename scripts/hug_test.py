from acnetwork import *
from baxter_env import *
import algo

import rospy
import os
import IPython
import numpy as np
import time
import copy

import pickle
import argparse


parser = argparse.ArgumentParser(description='AC')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                   help='random seed (default: 1)')
parser.add_argument('--resume', type=int, default=0, metavar='R',
                    help='if resume from previous model (default: No)')
parser.add_argument('--lstm', action='store_true', default=True,
                    help='if use LSTM (default: Yes)')
parser.add_argument('--moveit', action='store_true', default=False,
                    help='if use moveit (default: No)')
parser.add_argument('--test', action='store_true', default=False,
                    help='training or testing? (default: False)')
parser.add_argument('--step', type=int, default=10,
                    help='Baxter actions step for one episode (default: 10)')
parser.add_argument('--algo', default="ppo",
                    help='algorithm to use: a2c | ppo')
parser.add_argument('--clip', type=float, default=0.1,
                    help='ppo clipping parameter (default: 0.1)')
parser.add_argument('--tau', type=float, default=1.0,
                    help='gae parameter (default: 1.0)')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor for rewards (default: 0.99)')


if __name__ == '__main__':
    args = parser.parse_args(rospy.myargv()[1:])

    torch.manual_seed(args.seed)
    resume = args.resume  # whether load previous model, default 0
    use_lstm = args.lstm    # whether use LSTM structure, default false
    use_moveit = args.moveit  # whether use moveit to execute action, default false
    use_cuda = True
    TEST = args.test

    # Initilize ros environment, baxter agent
    rospy.init_node('baxter_hug')
    env = Baxter(use_moveit)
    state, _, _ = env.getstate()

    # Initialize a2c network
    actor_critic = ACNet(state, use_cuda, use_lstm)
    if use_cuda:
        actor_critic.cuda()

    # Initialize learning agent
    if args.algo == "a2c":
        agent = algo.A2Cagent(actor_critic, lr=1e-4, tau=args.tau, gamma=args.gamma)
    elif args.algo == "ppo":
        agent = algo.PPOagent(actor_critic, lr=1e-4, ppo_epoch=1, clip_param=args.clip, tau=args.tau, gamma=args.gamma)
    rollouts = Rollouts()
    buffer = Buffer()

    # Save the training models
    model_path = "./model_baxter_net/"
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # Record training information
    record_path = "./training_record/"
    if not os.path.exists(record_path):
        os.makedirs(record_path)
    Record = []
    Evaluation = []

    # Resume from models before
    if resume:
        print('Loading Model...')
        agent.actor_critic.network.load_state_dict(torch.load(model_path + "model-" + str(resume) + ".pt"))
        episode_num = resume
        file_load = open(model_path + 'triangulation.pkl', 'rb')
        tri_load = pickle.load(file_load)
        file_load.close()
        env.triangulation = tri_load
    else:
        episode_num = 0

    collision = 0
    while not rospy.is_shutdown() and episode_num <= 500:
        env.reset(episode_num, collision)
        # Calculate writhe before this episode
        _, w, collision = env.reward_evaluation(0, 0)
        max_w = w
        done = False
        if use_lstm:
            agent.actor_critic.cx = Variable(torch.zeros(1, agent.actor_critic.lstm_size))
            agent.actor_critic.hx = Variable(torch.zeros(1, agent.actor_critic.lstm_size))
            if use_cuda:
                agent.actor_critic.cx = agent.actor_critic.cx.cuda()
                agent.actor_critic.hx = agent.actor_critic.hx.cuda()

        for step in range(1, args.step + 1):
            state, writhe, InterMesh = env.getstate()

            with torch.no_grad():
                value, action, action_log_prob, action_entropy = \
                    agent.actor_critic.act(state, TEST=True)

            env.act(action.cpu().numpy().squeeze())
            reward, w, collision = env.reward_evaluation(w, step)
            max_w = max(max_w, w)

            if collision == 1:
                print "collision!!!!!!!!!!!!!!!!!!!!!!!"
                done = True
                break

            if collision == -1:
                print "target load error!!!!!!!!!!!!!!!!!!!!!"
                break

        if collision == -1 or 1:
            episode_num -= 1
            continue

        evaluation = [w, max_w]
        print("w:%f, max_w:%f" % (w, max_w))
        Evaluation.append(copy.deepcopy(evaluation))

        if episode_num % 100 == 0:
            w_count = 0
            maxw_count = 0
            eva = np.asarray(Evaluation)
            for i in range(len(eva)):
                if eva[i][0] > 1.5:
                    w_count += 1
                if eva[i][1] > 1.5:
                    maxw_count += 1
            rw = w_count / 100.0
            rmaxw = maxw_count / 100.0
            print("rw:", rw)
            print("rmaxw:", rmaxw)

            file_save2 = open(record_path + 'test.pkl', 'wb')
            pickle.dump(Evaluation, file_save2)
            file_save2.close()




