# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import random

##################### Parameters ########################

#Bound of the elements of the input matrices i.e. those should be in [0,...,B]
B = 50

#Input matrix size - A: s by r, B: s by t
s = 8000
r = 8000
t = 8000

print "Generating A,B %dx%d, B=%d" % (s, s, B)

#Generate and store matrices
A=np.matrix(np.random.random_integers(0,B,(s,r)))
B=np.matrix(np.random.random_integers(0,B,(s,t)))

print "A,B have been generated"

np.save('A', A)
np.save('B', B)

print "A,B have been stored"
