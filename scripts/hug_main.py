from acnetwork import *
from baxter_env import *

import sys
import rospy
import os
import IPython
import numpy as np
import time

import argparse




parser = argparse.ArgumentParser(description='A2C')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                   help='random seed (default: 1)')
parser.add_argument('--resume', type=bool, default=False, metavar='R',
                    help='if resume from previous model (default: No)')


if __name__ == '__main__':
    args = parser.parse_args(rospy.myargv()[1:])

    torch.manual_seed(args.seed)
    resume = args.resume  # whether load previous model

    use_cuda = True

    # Initilize ros environment, baxter agent
    rospy.init_node('baxter_hug')
    env = Baxter()

    # Initialize a2c network
    actor_critic = ACNet(use_cuda)
    if use_cuda:
        actor_critic.cuda()

    # Initialize learning agent
    agent = A2Cagent(actor_critic, lr = 1e-4)
    rollouts = Rollouts()

    # Begin to work!
    env.reset()
    done = False

    model_path = "./model_baxter_net/"
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # Resume from models before
    if resume:
        print ('Loading Model...')
        agent.actor_critic.network.load_state_dict(torch.load(model_path + "model-200.pt"))
        # target_net.load_state_dict(torch.load(model_path + "model_t-5000.pt"))
        episode_num = 200
        # OBSERVE = 5100
    else:
        episode_num = 0


    while not rospy.is_shutdown():

        episode_num += 1
        agent.actor_critic.cx = Variable(torch.zeros(1, agent.actor_critic.lstm_size))
        agent.actor_critic.hx = Variable(torch.zeros(1, agent.actor_critic.lstm_size))
        if use_cuda:
            agent.actor_critic.cx = agent.actor_critic.cx.cuda()
            agent.actor_critic.hx = agent.actor_critic.hx.cuda()

        # Calculate writhe before this episode
        _, w = env.reward_evaluation(0)

        for step in range(50):
            print "episode: %d, step:%d" % (episode_num, step+1)
            state = env.getstate()
            value, action, action_log_prob, action_entropy, (agent.actor_critic.hx, agent.actor_critic.cx) = \
                agent.actor_critic.act(state, (agent.actor_critic.hx, agent.actor_critic.cx))
            print "action:",
            print action

            env.act(0.2*action.cpu().numpy().squeeze())
            reward, w = env.reward_evaluation(w)
            print "reward:%f" % reward,
            print "w:%f" % w

            rollouts.insert(state, action, action_log_prob, action_entropy, reward, value)

            if done:
                break

        value_terminal = torch.zeros(1, 1)
        if not done:
            value_terminal = agent.actor_critic.getvalue(state, (agent.actor_critic.hx, agent.actor_critic.cx))
        rollouts.values.append(value_terminal)
        agent.update(rollouts)

        rollouts.clear()

        if episode_num % 200 == 0:
            print "saving model..."
            torch.save(agent.actor_critic.network.state_dict(), model_path + 'model-' + str(episode_num) + '.pt')
            # torch.save(target_net.state_dict(), model_path + 'model_t-' + str(episode_num) + '.pt')







