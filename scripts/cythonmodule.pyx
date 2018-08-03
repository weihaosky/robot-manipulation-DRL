import numpy as np
import math

cimport numpy as np
cimport cython

cdef extern from "complex.h":
    double complex csqrt(double complex z) nogil
    double complex cexp(double complex z) nogil
    double creal(double complex z) nogil
    double cimag(double complex z) nogil

from libc.math cimport sqrt, pow
from scipy.integrate import dblquad


cdef class Params:
    cdef public double x1, y1, z1, x2, y2, z2, a1, b1, c1, a2, b2, c2, Prod0, Prod1, Prod2

    def __init__(self, x1, y1, z1, x2, y2, z2, a1, b1, c1, a2, b2, c2, Prod0, Prod1, Prod2):
        self.x1 = x1
        self.y1 = y1
        self.z1 = z1
        self.x2 = x2
        self.y2 = y2
        self.z2 = z2
        self.a1 = a1
        self.b1 = b1
        self.c1 = c1
        self.a2 = a2
        self.b2 = b2
        self.c2 = c2
        self.Prod0 = Prod0
        self.Prod1 = Prod1
        self.Prod2 = Prod2


@cython.cdivision(True)
def inte(double s, double t, Params p):

    x = p.x1 + (p.x2 - p.x1) * s
    y = p.y1 + (p.y2 - p.y1) * s
    z = p.z1 + (p.z2 - p.z1) * s
    a = p.a1 + (p.a2 - p.a1) * t
    b = p.b1 + (p.b2 - p.b1) * t
    c = p.c1 + (p.c2 - p.c1) * t
    intergral = (p.Prod0*(x-a) + p.Prod1*(y-b) + p.Prod2*(z-c)) / pow((x-a)**2 + (y-b)**2 + (z-c)**2, 3.0/2)
    return intergral

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

    p = Params(x1 = x1,
               y1 = y1,
               z1 = z1,
               x2 = x2,
               y2 = y2,
               z2 = z2,
               a1 = a1,
               b1 = b1,
               c1 = c1,
               a2 = a2,
               b2 = b2,
               c2 = c2,
               Prod0 = Prod[0],
               Prod1 = Prod[1],
               Prod2 = Prod[2]
               )

    result = dblquad( inte , 0, 1, lambda t:0, lambda t:1, args=(p,))
    gli = result[0] / (4*math.pi)
    error = result[1] / (4*math.pi)
    return (gli, error)





