import torch
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
import IPython


class A2Cagent(object):
    def __init__(self,
                 actor_critic,
                 lr = 1e-4,
                 gamma=0.99,
                 tau=1.0,
                 use_cuda = True):
        self.gamma = gamma   # discounting factor
        self.tau = tau   # used for calculating GAE

        self.actor_critic = actor_critic
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01

        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr)
        self.use_cuda = self.actor_critic.use_cuda

    def update(self, rollouts):

        states_batch = rollouts.states
        actions_batch = rollouts.actions
        rewards_batch = Variable(torch.from_numpy(np.asarray(rollouts.rewards))).float()
        if self.use_cuda:
            rewards_batch = rewards_batch.cuda()
        values_batch = rollouts.values
        action_log_probs_batch = rollouts.action_log_probs
        entropies_batch = rollouts.entropies
        memory_batch = rollouts.memory

        values, action_log_probs, a_dist_entropies, memory = self.actor_critic.evaluate_actions(states_batch,
                                                                                                actions_batch,
                                                                                                memory_batch)

        action_loss = 0
        value_loss = 0
        gae = Variable(torch.zeros(1, 1))
        R = Variable(torch.zeros(1,1))
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

            action_loss = action_loss - \
                          (action_log_probs[i] * gae - self.entropy_coef * a_dist_entropies[i]).sum()

        loss = value_loss * self.value_loss_coef + action_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 40)
        self.optimizer.step()

        return loss


