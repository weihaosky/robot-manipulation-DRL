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


class PPOagent(object):
    def __init__(self,
                 actor_critic,
                 lr=1e-4,
                 ppo_epoch=1,
                 clip_param=0.1,
                 use_cuda=True):
        self.gamma = 0.99  # discounting factor
        self.tau = 1.0  # used for calculating GAE
        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch

        self.actor_critic = actor_critic
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01

        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr)
        self.use_cuda = self.actor_critic.use_cuda

    def update(self, buffer):

        for epoch in range(self.ppo_epoch):
            data_generator = buffer.generator()

            for rollouts in data_generator:
                states_batch = rollouts.states
                actions_batch = rollouts.actions
                rewards_batch = Variable(torch.from_numpy(np.asarray(rollouts.rewards))).float()
                if self.use_cuda:
                    rewards_batch = rewards_batch.cuda()
                values_batch = rollouts.values
                old_action_log_probs_batch = rollouts.action_log_probs
                old_action_entropies_batch = rollouts.entropies
                memory_batch = rollouts.memory

                values, action_log_probs, a_dist_entropies, memory = self.actor_critic.evaluate_actions(states_batch, actions_batch, memory_batch)

                action_loss = 0
                value_loss = 0
                gae = Variable(torch.zeros(1, 1))
                R = Variable(torch.zeros(1, 1))
                if self.use_cuda:
                    gae = gae.cuda()
                    R = R.cuda()

                for i in reversed(range(len(rewards_batch))):
                    R = self.gamma * R + rewards_batch[i]
                    advantage = R - values[i]
                    value_loss = value_loss + self.value_loss_coef * advantage.pow(2)

                    # Generalized Advantage Estimataion
                    delta_t = rewards_batch[i] + self.gamma * values[i + 1] - values[i]
                    gae = gae * self.gamma * self.tau + delta_t
                    ratio = torch.exp(action_log_probs[i] - old_action_log_probs_batch[i])
                    print("ratio:", ratio)
                    surr1 = ratio * gae
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * gae

                    action_loss = action_loss - \
                                  (torch.min(surr1, surr2) - self.entropy_coef * a_dist_entropies[i]).sum()

                loss = value_loss * self.value_loss_coef + action_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 40)
                self.optimizer.step()

        num_updates = self.ppo_epoch * len(buffer.roll)
        loss_epi = loss / num_updates
        print("update num:", num_updates)

        return loss_epi