import sys
import mosek
import mosek.fusion
from   mosek.fusion import *
from   mosek import LinAlg
from   math import sqrt

def vec(e):
    """
    Assuming that e is an NxN expression, return the lower triangular part as a vector.
    """
    N = e.getShape().dim(0)
    
    rows = [i for i in range(N) for j in range(i,N)]
    cols = [j for i in range(N) for j in range(i,N)]
    vals = [ 2.0**0.5  if i!=j else 1.0 for i in range(N)  for j in range(i,N)]

    return Expr.flatten(Expr.mulElm(e, Matrix.sparse(N,N,rows,cols,vals)))
def nearestcorr(A):

    N = A.numRows()

    # Create a model with the name 'NearestCorrelation
    with Model("NearestCorrelation") as M:

        # Setting up the variables
        X = M.variable("X", Domain.inPSDCone(N))
        t = M.variable("t", 1, Domain.unbounded())

        # (t, vec (A-X)) \in Q
        M.constraint("C1", Expr.vstack(t, vec(Expr.sub(A,X))), Domain.inQCone() )

        # diag(X) = e
        M.constraint("C2",X.diag(), Domain.equalsTo(1.0))

        # Objective: Minimize t
        M.objective(ObjectiveSense.Minimize, t)
        M.solve()

        return X.level(),t.level()

if __name__ == '__main__':

    N = 5

    A = Matrix.dense(N,N,[ 0.0,  0.5,  -0.1,  -0.2,   0.5,
                          0.5,  1.25, -0.05, -0.1,   0.25,
                         -0.1, -0.05,  0.51,  0.02, -0.05,
                         -0.2, -0.1,   0.02,  0.54, -0.1,
                          0.5,  0.25, -0.05, -0.1,   1.25])
    
    X,t = nearestcorr(A)

    print("--- Nearest Correlation ---")
    print("X = ",X)
    print("t = ",t)
    
