import numpy
import math
import random
import sys

import mosek

from mosek.fusion import *

import gmmeancone

v = 100.0
for n in range(2,11):
    with Model("Testing") as M:
        t = M.variable('t', 1, Domain.unbounded())
        x = M.variable('x', n, Domain.unbounded())
        
        #(x[0]*...*x[n-1]) >= t^n, x,t>=0
        gmmeancone.appendcone(M,t,x)

        # x[0] <= v  
        M.constraint('xltv',x.index(0),Domain.lessThan(v)) 

        # x[i]=1.0, for i=1,...,n-1
        c = M.constraint('fixtoone',x.slice(1,n),Domain.equalsTo(1.0))

        print(c.toString())

        M.objective(ObjectiveSense.Maximize, t)

        M.writeTask('dump.opf')

        M.solve()

        print('Check %e %e' % (v**(1.0/n),t.level()))


