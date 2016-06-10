import numpy
import math
import random
import sys

import mosek

from mosek.fusion import *

import gmmeancone

def  buildandsolve(y):   # y[i+1]-y[i]>0 
    with Model("Max likelihood") as M:

        #M.setLogHandler(sys.stdout)  # Make sure we get some output

        n      = len(y)
   
        t      = M.variable('t', 1, Domain.unbounded())
        x      = M.variable('x', n, Domain.greaterThan(0.0))

        dy     = [y[i+1]-y[i] for i in range(0,n-1)]

        eleft  = Expr.mulElm(dy[1:n-1],x.slice(0,n-2))
        emid   = Expr.add(Expr.mulElm(dy[0:n-2],x.slice(1,n-1)),Expr.mulElm(dy[1:n-1],x.slice(1,n-1)))
        eright = Expr.mulElm(dy[0:n-2],x.slice(2,n))

        # Debug print: print(eleft.toString())

        M.constraint('convex',Expr.sub(Expr.sub(emid,eleft),eright),Domain.equalsTo(0.0))
        M.constraint('area',Expr.mul(0.5,Expr.dot(dy,Expr.add(x.slice(0,n-1),x.slice(1,n)))),Domain.equalsTo(1.0))

        gmmeancone.appendcone(M,t,x)

        M.objective(ObjectiveSense.Maximize, t)

        M.solve()

        return x.level()


