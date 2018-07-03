import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical, Normal


import math
import random
import numpy as np
from collections import deque
import IPython

from utils import init, init_normc_



class MLPBase(nn.Module):

    def __init__(self, state_shape, action_dim, lstm_size, use_cuda, use_lstm, name=''):
        super(MLPBase, self).__init__()

        init_ = lambda m: init(m,
                               init_normc_,
                               lambda x: nn.init.constant_(x, 0))

        self.state_shape = state_shape
        self.action_dim = action_dim
        self.lstm_size = lstm_size
        self.use_cuda = use_cuda
        self.use_lstm = use_lstm
        # self.conv1 = nn.Conv2d(self.state_shape[0], 64, kernel_size=9, stride=1, padding = 4)
        # # self.bn1 = nn.BatchNorm2d(64)
        # self.conv2 = nn.Conv2d(64, 64, kernel_size=7, stride=1, padding = 3)
        # # self.bn2 = nn.BatchNorm2d(64)
        # self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding = 2)
        # # self.bn3 = nn.BatchNorm2d(128)
        # self.conv4 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding = 2)
        # # self.bn4 = nn.BatchNorm2d(128)
        # self.conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding = 1)
        # # self.bn5 = nn.BatchNorm2d(128)
        # self.conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding = 1)
        # self.max4 = nn.MaxPool2d(2)
        # # self.bn6 = nn.BatchNorm2d(128)
        
        self.hidden11 = init_(nn.Linear(self.state_shape[3], 512))
        self.hidden12 = init_(nn.Linear(self.state_shape[4], 512))
        # self.hidden13 = init_(nn.Linear(self.state_shape[0], 256))
        # self.hidden14 = init_(nn.Linear(self.state_shape[3], 256))
        self.hidden2 = init_(nn.Linear(1024, 512))
        if self.use_lstm:
            self.lstm = nn.LSTMCell(512, self.lstm_size)
            self.action_head = init_(nn.Linear(self.lstm_size, self.action_dim))
            self.action_sigma = init_(nn.Linear(self.lstm_size, self.action_dim))
            self.value_head = init_(nn.Linear(self.lstm_size, 1))

            self.lstm.bias_ih.data.fill_(0)
            self.lstm.bias_hh.data.fill_(0)
        else:
            self.action_head = init_(nn.Linear(512, self.action_dim))
            self.action_sigma = init_(nn.Linear(512, self.action_dim))
            self.value_head = init_(nn.Linear(512, 1))


    def forward(self, state, (hx, cx)):
        # x = x.float().div(255.0)
        # x = F.relu(self.conv1(x))
        # x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        # x = F.relu(self.conv3(x))
        # x = F.max_pool2d(F.relu(self.conv4(x)), (2, 2))
        # x = F.max_pool2d(F.relu(self.conv5(x)), (2, 2))
        # x = F.relu(self.conv6(x))
        # x = self.max4(x)
        # x = x.view(-1, 8*8*128)
        # x, (hx,cx) = inputs
        x1 = Variable(torch.from_numpy(state[3])).float()
        x2 = Variable(torch.from_numpy(state[4])).float()
        # x3 = Variable(torch.from_numpy(state[0])).float()
        # x4 = Variable(torch.from_numpy(state[3])).float()
        if self.use_cuda:
            x1 = x1.cuda()
            x2 = x2.cuda()
            # x3 = x3.cuda()
            # x4 = x4.cuda()
        x1 = F.relu(self.hidden11(x1))
        x2 = F.relu(self.hidden12(x2))
        # x3 = F.relu(self.hidden13(x3))
        # x4 = F.relu(self.hidden14(x4))
        # x = torch.cat((x1, x2, x3, x4), -1)
        x = torch.cat((x1, x2), -1)
        x = F.relu(self.hidden2(x))
        x = x.view(-1, 512)
        if self.use_lstm:
            hx, cx = self.lstm(x, (hx, cx))
            x = hx
        c = self.value_head(x)
        a = F.tanh(self.action_head(x))
        a_sigma = F.softplus(self.action_sigma(x))
        return c, a, a_sigma, (hx, cx)


class ACNet(nn.Module):
    def __init__(self, state, use_cuda, use_lstm):
        super(ACNet, self).__init__()
        self.use_cuda = use_cuda
        self.use_lstm = use_lstm
        self.lstm_size = 64
        self.state_shape = [len(state[0]), len(state[1]), len(state[2]), len(state[3]), len(state[4])]
        self.action_dim = 7
        self.network = MLPBase(state_shape=self.state_shape, action_dim=self.action_dim,
                               lstm_size=self.lstm_size, use_lstm=self.use_lstm, use_cuda=self.use_cuda)
        self.cx = Variable(torch.zeros(1, self.lstm_size))
        self.hx = Variable(torch.zeros(1, self.lstm_size))
        if self.use_cuda:
            self.cx = self.cx.cuda()
            self.hx = self.hx.cuda()


    def act(self, states):
        # states = Variable(torch.from_numpy(states))
        # if self.use_cuda:
        #     states = states.cuda()
        value, action_mu, action_sigma, (self.hx, self.cx) = self.network(states, (self.hx, self.cx))
        a_dist = Normal(action_mu, action_sigma/10.0)
        action = a_dist.sample()
        a_log_probs = a_dist.log_prob(action)
        a_dist_entropy = a_dist.entropy()

        print "action_mu:",
        print action_mu
        print "action_sigma:",
        print action_sigma
        # print "value:",
        # print value

        return value, action, a_log_probs, a_dist_entropy

    def getvalue(self, states):
        # states = Variable(torch.from_numpy(states))
        # if self.use_cuda:
        #     states = states.cuda()
        value, _, _, _ = self.network(states, (self.hx, self.cx))
        return value

    def evaluate_actions(self, states, action):
        value, action_mu, action_sigma, (self.hx, self.cx) = self.network(states, (self.hx, self.cx))
        a_dist = Normal(action_mu, action_sigma)

        a_log_probs = a_dist.log_prob(action)
        a_dist_entropy = a_dist.entropy().mean()

        return value, a_log_probs, a_dist_entropy


class A2Cagent(object):
    def __init__(self,
                 actor_critic,
                 lr = 1e-4,
                 use_cuda = True):
        self.gamma = 0.99   # discounting factor
        self.tau = 1.0   # used for calculating GAE

        self.actor_critic = actor_critic
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01

        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr)
        self.use_cuda = self.actor_critic.use_cuda

    def update(self, rollouts):

        states = rollouts.states
        actions = rollouts.actions
        rewards = Variable(torch.from_numpy(np.asarray(rollouts.rewards))).float()
        if self.use_cuda:
            rewards = rewards.cuda()
        values = rollouts.values
        action_log_probs = rollouts.action_log_probs
        entropies = rollouts.entropies

        action_loss = 0
        value_loss = 0
        gae = Variable(torch.zeros(1, 1))
        R = Variable(torch.zeros(1,1))
        if self.use_cuda:
            gae = gae.cuda()
            R = R.cuda()

        for i in reversed(range(len(rewards))):
            R = self.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + self.value_loss_coef * advantage.pow(2)

            # Generalized Advantage Estimataion
            delta_t = rewards[i] + self.gamma * values[i + 1] - values[i]
            gae = gae * self.gamma * self.tau + delta_t

            action_loss = action_loss - \
                          (action_log_probs[i] * gae - self.entropy_coef * entropies[i]).sum()


        loss = value_loss * self.value_loss_coef + action_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 40)
        self.optimizer.step()


class Rollouts(object):
    def __init__(self):
        self.states = []
        self.actions = []
        self.action_log_probs = []
        self.entropies = []
        self.rewards = []
        self.values = []


    def insert(self, state, action, action_log_prob, entropy, reward, value):
        self.states.append(state)
        self.actions.append(action)
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(reward)
        self.entropies.append(entropy)
        self.values.append(value)

    def clear(self):
        self.states = []
        self.actions = []
        self.action_log_probs = []
        self.rewards = []
        self.entropies = []
        self.values = []










class ReplayBuffer():
    REPLAY_MEMORY = 150000 # number of previous transitions to remember
    def __init__(self):
        self.replayMemory = deque()
    def setPerception(self, currentState, action, reward, nextState, terminal):
        self.replayMemory.append((currentState, action, reward, nextState, terminal))
        if len(self.replayMemory) > self.REPLAY_MEMORY:
            self.replayMemory.popleft()


def train(T_net, P_net, buffer):
    BATCH_SIZE = 32
    GAMMA = 0.99
    optimizer = optim.Adam(P_net.parameters(), lr=1e-4)

    if len(buffer.replayMemory) < BATCH_SIZE:
        return
    # Step 1: obtain random minibatch from replay memory
    minibatch = random.sample(buffer.replayMemory, BATCH_SIZE)
    state_batch = Variable(torch.from_numpy(np.asarray([data[0] for data in minibatch]).transpose((0, 3, 1, 2))).cuda())
    ac_batch = Variable(torch.from_numpy(np.asarray([data[1] for data in minibatch])).cuda())
    reward_batch = Variable(torch.from_numpy(np.asarray([data[2] for data in minibatch])).cuda()).float()
    nextState_batch = Variable(torch.from_numpy(np.asarray([data[3] for data in minibatch]).transpose((0, 3, 1, 2))).cuda())

    # Step 2: calculate y
    nextState_batch.volatile = True
    value_next_batch = T_net(nextState_batch)[0]
    targetQ_batch = reward_batch[0].float() + GAMMA * value_next_batch[0]
    for i in range(1, BATCH_SIZE):
        targetQ_batch = torch.cat((targetQ_batch, reward_batch[i].float() + GAMMA * value_next_batch[i]), 0)

    value_head = P_net(state_batch)[0]
    targetQ_batch.volatile = False
    value_loss = F.mse_loss(value_head, targetQ_batch, size_average=False)

    action_head = T_net(state_batch)[1]
    aprob_batch = action_head[0][ac_batch[0]]
    for i in range(1, BATCH_SIZE):
        aprob_batch = torch.cat((aprob_batch, action_head[i][ac_batch[i]]), 0)

    policy_loss = -torch.log(aprob_batch) * (targetQ_batch - value_head.view(32))
    policy_loss = policy_loss.sum()

    entropy = (-aprob_batch * torch.log(aprob_batch)).sum()

    print "v loss: ",
    print value_loss,
    print "p loss: ",
    print policy_loss

    loss = value_loss + policy_loss - 0.01 * entropy

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # for param in Primary_DQN.parameters():
    #     param.grad.data.clamp_(-1, 1)
    optimizer.step()

    return loss


def updateTarget(Target_DQN, Primary_DQN):
    TAU = 0.001
    for (W_p, W_t) in zip(Primary_DQN.parameters(), Target_DQN.parameters()):
        W_t.data = W_t.data * (1-TAU) + W_p.data * TAU


def select_action(x, target_net, episode_number, Distance, Block_position_last, Target_x):
    x = Variable(torch.from_numpy(x.transpose((0, 3, 1, 2))).cuda(), volatile=True)
    # x = Variable(torch.from_numpy(x).cuda(), volatile=True)
    value, actions = target_net(x)
    value = value.cpu().data[0].numpy()
    action = actions.multinomial().cpu().data[0].numpy()
    aprob = actions.cpu().data[0].numpy()[action]


    return action, actions, aprob, value

