import torch
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
import IPython


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

        return loss


