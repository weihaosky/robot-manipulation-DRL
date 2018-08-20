from acnetwork import *
from baxter_env import *
import algo

import sys
import rospy
import os
import IPython
import numpy as np
import time
import copy

import pickle
import argparse
import matplotlib.pyplot as plt
import seaborn as sns


parser = argparse.ArgumentParser(description='AC')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                   help='random seed (default: 1)')
parser.add_argument('--resume', type=int, default=0, metavar='R',
                    help='if resume from previous model (default: No)')
parser.add_argument('--lstm', action='store_false', default=True,
                    help='if use LSTM (default: Yes)')
parser.add_argument('--moveit', action='store_true', default=False,
                    help='if use moveit (default: No)')
parser.add_argument('--test', action='store_true', default=False,
                    help='training or testing? (default: False)')
parser.add_argument('--step', type=int, default=10,
                    help='Baxter actions step for one episode (default: 10)')
parser.add_argument('--algo', default="ppo",
                    help='algorithm to use: a2c | ppo')
parser.add_argument('--clip', type=float, default=0.2,
                    help='ppo clipping parameter (default: 0.2)')
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

    # Begin to work!
    # env.reset()
    # env.right_limb_interface.set_joint_position_speed(0.9)
    # done = False
    # Remove models from the scene on shutdown
    # rospy.on_shutdown(env.clear)


    # Resume from models before
    if resume:
        print('Loading Model...')
        agent.actor_critic.network.load_state_dict(torch.load(model_path + "model-" + str(resume) + ".pt"))
        # target_net.load_state_dict(torch.load(model_path + "model_t-5000.pt"))
        episode_num = resume
        # OBSERVE = 5100
        file_load = open(record_path + 'triangulation.pkl', 'rb')
        tri_load = pickle.load(file_load)
        file_load.close()
        env.triangulation = tri_load
    else:
        episode_num = 0
        file_save = open(record_path + 'triangulation.pkl', 'wb')
        pickle.dump(env.triangulation, file_save)
        file_save.close()

    collision = 0
    while not rospy.is_shutdown() and episode_num <= 2000:

        start_time = time.time()    # timing for one episode
        episode_num += 1
        env.reset(episode_num, collision)
        done = False
        if use_lstm:
            agent.actor_critic.cx = Variable(torch.zeros(1, agent.actor_critic.lstm_size))
            agent.actor_critic.hx = Variable(torch.zeros(1, agent.actor_critic.lstm_size))
            if use_cuda:
                agent.actor_critic.cx = agent.actor_critic.cx.cuda()
                agent.actor_critic.hx = agent.actor_critic.hx.cuda()

        # Calculate writhe before this episode
        _, w, collision = env.reward_evaluation(0, 0)
        print("Starting w:%f" % w)
        max_w = w
        # store the hxcx before the 1st step
        rollouts.memory.append(copy.deepcopy((agent.actor_critic.hx, agent.actor_critic.cx)))

        for step in range(1, args.step+1):
            print "----------- episode: %d, step:%d ------------" % (episode_num, step)
            state, writhe, InterMesh = env.getstate()

            # # heat_map evolution
            # fig = plt.figure(0)
            # ax = fig.add_subplot(121)
            # sns.heatmap(writhe, vmax=0.02, vmin=-0.02, cmap=plt.cm.hot)
            # # plt.colorbar(ax.imshow(writhe, cmap=matplotlib.cm.hot), norm=matplotlib.colors.Normalize(vmin=-1, vmax=1), ticks=[-1, 0, 1])
            # ax = fig.add_subplot(122)
            # # plt.colorbar(ax.imshow(InterMesh, cmap=matplotlib.cm.hot))
            # sns.heatmap(InterMesh, vmax=0.5, vmin=-0.5, cmap=plt.cm.hot)
            # # plt.colorbar()
            # plt.savefig("state_heat.png")
            # IPython.embed()
            # plt.clf()


            with torch.no_grad():
                value, action, action_log_prob, action_entropy = \
                    agent.actor_critic.act(state, TEST)
                # print("action:", action.data)

            # time_t = time.time()
            env.act(action.cpu().numpy().squeeze())
            # print("time for env.act: ", time.time() - time_t)

            reward, w, collision = env.reward_evaluation(w, step)
            print("reward:%f" % reward, "w:%f" % w)
            max_w = max(max_w, w)

            rollouts.insert(state, action, action_log_prob, action_entropy, reward, value, (agent.actor_critic.hx, agent.actor_critic.cx))

            if collision == 1:
                print "collision!!!!!!!!!!!!!!!!!!!!!!!"
                done = True
                break

            if collision == -1:
                print "target load error!!!!!!!!!!!!!!!!!!!!!"
                break

        if collision == -1:
            episode_num -= 1
            rollouts.clear()
            continue

        value_terminal = torch.zeros(1, 1)
        if use_cuda:
            value_terminal = value_terminal.cuda()
        state, _, _ = env.getstate()
        if not done:
            with torch.no_grad():
                value_terminal = agent.actor_critic.getvalue(state)
        rollouts.states.append(copy.deepcopy(state))
        rollouts.values.append(copy.deepcopy(value_terminal))

        if not TEST:    # training
            if args.algo == "a2c":
                loss = agent.update(rollouts)
            if args.algo == "ppo":
                buffer.insert(rollouts)
                loss = agent.update(buffer)
            loss_record = loss.cpu().detach().numpy().squeeze()
        else:
            loss_record = 0.0

        # record the training information for analysis
        reward_mean = np.asarray(rollouts.rewards).mean()
        values = []
        for item in rollouts.values:
            values.append(item.cpu().detach().numpy())
        value_mean = np.asarray(values).mean()
        time_epi = time.time() - start_time
        record = [reward_mean, value_mean, w, max_w, loss_record, time_epi]
        print("episode %d cost time: %fs" % (episode_num, time_epi))
        print("record:", record)
        Record.append(copy.deepcopy(record))

        rollouts.clear()

        # evaluation
        if episode_num % 10 == 0:
            print("--------------- Evaluation: ---------------")
            collision = 2
            while collision != 0:
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
            evaluation = [w, max_w]
            print("w:%f, max_w:%f" % (w, max_w))
            Evaluation.append(copy.deepcopy(evaluation))


        if episode_num % 100 == 0:
            file_save = open(record_path + 'Record.pkl', 'wb')
            pickle.dump(Record, file_save)
            file_save.close()

            file_save2 = open(record_path + 'Evaluation.pkl', 'wb')
            pickle.dump(Evaluation, file_save2)
            file_save2.close()


        if episode_num % 100 == 0:
            print "saving model..."
            torch.save(agent.actor_critic.network.state_dict(), model_path + 'model-' + str(episode_num) + '.pt')
            # torch.save(target_net.state_dict(), model_path + 'model_t-' + str(episode_num) + '.pt')




