import torch
import torch.nn as nn

import numpy as np
import math
import threading
from scipy.integrate import quad, dblquad, nquad


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


# https://github.com/openai/baselines/blob/master/baselines/common/tf_util.py#L87
def init_normc_(weight, gain=1):
    weight.normal_(0, 1)
    weight *= gain / torch.sqrt(weight.pow(2).sum(1, keepdim=True))


# calculate gussian linking integral for two lines X1--X2 and Y1--Y2
def GLI(X1=(0.0, 0.0, 0.0), X2 = (1.0, 0.0, 1.0), Y1 = (0.0, 0.0, 1.0), Y2 = (0.0, 1.0, 1.0)):
    x1 = X1[0]
    y1 = X1[1]
    z1 = X1[2]
    x2 = X2[0]
    y2 = X2[1]
    z2 = X2[2]

    a1 = Y1[0]
    b1 = Y1[1]
    c1 = Y1[2]
    a2 = Y2[0]
    b2 = Y2[1]
    c2 = Y2[2]

    D1 = np.array([x2-x1, y2-y1, z2-z1])
    D2 = np.array([a2-a1, b2-b1, c2-c1])
    Prod = np.cross(D1, D2)

    def inte(s, t):
        x = x1 + (x2 - x1) * s
        y = y1 + (y2 - y1) * s
        z = z1 + (z2 - z1) * s
        a = a1 + (a2 - a1) * t
        b = b1 + (b2 - b1) * t
        c = c1 + (c2 - c1) * t
        return (Prod[0]*(x-a) + Prod[1]*(y-b) + Prod[2]*(z-c)) / np.power((x-a)**2 + (y-b)**2 + (z-c)**2, 3.0/2)

    result = dblquad( inte , 0, 1, lambda t:0, lambda t:1)
    gli = result[0] / (4*math.pi)
    error = result[1] / (4*math.pi)
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


