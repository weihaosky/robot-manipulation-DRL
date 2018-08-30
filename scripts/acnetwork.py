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
import copy
import IPython

from utils import init, init_normc_, AddBias


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
        self.stddev = 1
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

        self.logstd = AddBias(torch.zeros(action_dim))

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

        if self.stddev == 1:
            a_sigma = F.softplus(self.action_sigma(x))
        else:
            zeros = torch.zeros(a.size())
            if x.is_cuda:
                zeros = zeros.cuda()
            action_logstd = self.logstd(zeros)
            a_std = action_logstd.exp()
            a_sigma = a_std

        return c, a, a_sigma, (hx, cx)


class ACNet(nn.Module):
    def __init__(self, state, use_cuda, use_lstm):
        super(ACNet, self).__init__()
        self.use_cuda = use_cuda
        self.use_lstm = use_lstm
        self.lstm_size = 64
        self.state_shape = [len(state[0]), len(state[1]), len(state[2]), len(state[3]), len(state[4])]
        self.action_dim = 14
        self.network = MLPBase(state_shape=self.state_shape, action_dim=self.action_dim,
                               lstm_size=self.lstm_size, use_lstm=self.use_lstm, use_cuda=self.use_cuda)
        self.cx = Variable(torch.zeros(1, self.lstm_size))
        self.hx = Variable(torch.zeros(1, self.lstm_size))
        if self.use_cuda:
            self.cx = self.cx.cuda()
            self.hx = self.hx.cuda()

    def act(self, states, TEST):
        # states = Variable(torch.from_numpy(states))
        # if self.use_cuda:
        #     states = states.cuda()
        value, action_mu, action_sigma, (self.hx, self.cx) = self.network(states, (self.hx, self.cx))
        a_dist = Normal(action_mu, action_sigma)
        if not TEST:
            action = a_dist.sample()
        else:
            action = action_mu
        a_log_probs = a_dist.log_prob(action)
        a_dist_entropy = a_dist.entropy()

        # print("action_mu:", action_mu)
        # print("action_sigma:", action_sigma.data)
        # print("action:", action)
        # print("hx,cx:",self.hx,self.cx)
        # print "value:",
        # print value

        return value, action, a_log_probs, a_dist_entropy

    def getvalue(self, states):
        # states = Variable(torch.from_numpy(states))
        # if self.use_cuda:
        #     states = states.cuda()
        value, _, _, _ = self.network(states, (self.hx, self.cx))
        return value

    def evaluate_actions(self, states, action, memory):
        hx = [hxcx[0] for hxcx in memory]
        cx = [hxcx[1] for hxcx in memory]
        len_batch = len(action)
        # print("len_batch:", len_batch)
        value = [None] * (len_batch + 1)
        action_mu = [None] * len_batch
        action_sigma = [None] * len_batch
        a_dist = [None] * len_batch
        a_log_probs = [None] * len_batch
        a_dist_entropy = [None] * len_batch

        for i in range(len_batch):
            value[i], action_mu[i], action_sigma[i], (hx[i], cx[i]) = self.network(states[i], (hx[i], cx[i]))
            a_dist[i] = Normal(action_mu[i], action_sigma[i])
            a_log_probs[i] = a_dist[i].log_prob(action[i])
            a_dist_entropy[i] = a_dist[i].entropy()
        value[i+1], _, _, _ = self.network(states[i+1], (hx[i+1], cx[i+1]))

        return value, a_log_probs, a_dist_entropy, (hx, cx)


class Rollouts(object):
    def __init__(self):
        self.states = []
        self.actions = []
        self.action_log_probs = []
        self.entropies = []
        self.rewards = []
        self.values = []
        self.memory = []

    def insert(self, state, action, action_log_prob, entropy, reward, value, memory):
        self.states.append(copy.deepcopy(state))
        self.actions.append(copy.deepcopy(action))
        self.action_log_probs.append(copy.deepcopy(action_log_prob))
        self.rewards.append(copy.deepcopy(reward))
        self.entropies.append(copy.deepcopy(entropy))
        self.values.append(copy.deepcopy(value))
        self.memory.append(copy.deepcopy(memory))

    def clear(self):
        self.states = []
        self.actions = []
        self.action_log_probs = []
        self.rewards = []
        self.entropies = []
        self.values = []
        self.memory = []


class Buffer(object):
    def __init__(self):
        self.size = 4
        self.roll = []

    def insert(self, rollouts):
        if len(self.roll) >= 4:
            self.roll.pop(0)
        self.roll.append(copy.deepcopy(rollouts))

    def generator(self):
        perm = torch.randperm(len(self.roll))
        for i in range(len(self.roll)):
            # print("replay order:", i)
            yield self.roll[i]

    def clear(self):
        self.roll = []