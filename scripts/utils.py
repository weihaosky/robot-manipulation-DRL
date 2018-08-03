import torch
import torch.nn as nn

import numpy as np
import math
import threading
from scipy.integrate import quad, dblquad, nquad
import scipy.signal

import pyximport; pyximport.install(reload_support=True)
import cythonmodule

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def init_normc_(weight, gain=1):
    weight.normal_(0, 1)
    weight *= gain / torch.sqrt(weight.pow(2).sum(1, keepdim=True))


class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


# calculate gussian linking integral for two lines X1--X2 and Y1--Y2
def GLI(X1=(0.0, 0.0, 0.0), X2 = (1.0, 0.0, 1.0), Y1 = (0.0, 0.0, 1.0), Y2 = (0.0, 1.0, 1.0)):
    gli, error = cythonmodule.GLI(X1 = X1, X2 = X2, Y1 = Y1, Y2 = Y2)
    return (gli, error)


class InfoGetter(object):
    def __init__(self):
        #event that will block until the info is received
        self._event = threading.Event()
        #attribute for storing the rx'd message
        self._msg = None

    def __call__(self, msg):
        #Uses __call__ so the object itself acts as the callback
        #save the data, trigger the event
        self._msg = msg
        self._event.set()

    def get_msg(self, timeout=0.5):
        """Blocks until the data is rx'd with optional timeout
        Returns the received message
        """
        self._event.wait(timeout)
        return self._msg


def find_neighbors(pindex, triang):
    return triang.vertex_neighbor_vertices[1][triang.vertex_neighbor_vertices[0][pindex]:triang.vertex_neighbor_vertices[0][pindex+1]]


def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


