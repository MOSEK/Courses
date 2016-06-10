import math

import mosek
from mosek.fusion import *

def appendcone(M,t,x):
    # t is scalar variable
    # x is n dimmensional variable

    lenx = x.size()  
    l    = 0
    while 2**l<lenx:
        l = l+1

    n    = 2**l
    d    = 2*n-1

    idx1 = range(1,d,2)
    idx2 = range(2,d,2)
    idx3 = range(0,d-n,1)

    g    = M.variable('g', d, Domain.unbounded())
    
    M.constraint('gm_RQs', Expr.hstack(g.pick(idx1),g.pick(idx2),g.pick(idx3)), Domain.inRotatedQCone())

    # t = sqrt(n)*g(0)
    M.constraint('gm_t', Expr.sub(Expr.mul(math.sqrt(n),t),g.index(0)), Domain.equalsTo(0.0))

    # Set leaf nodes equal to x.
    M.constraint('gm_g=x', Expr.sub(x,g.slice(d-n,d-n+lenx)), Domain.equalsTo(0.0))   	

    # Only the leaf nodes has to be psostive
    M.constraint('gm_t>=0', t, Domain.greaterThan(0.0))     
    M.constraint('gm_g>=0', g.slice(d-n,d-n+lenx), Domain.greaterThan(0.0))     

    if lenx<n:
        # Handle the uneven case
    	M.constraint('gm_rem', Expr.sub(g.slice(d-n+lenx,d),Expr.outer([1.0]*(n-lenx),t)), Domain.equalsTo(0.0))   	

