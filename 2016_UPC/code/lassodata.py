import random
import sys

def get(m,n,seed):

    random.seed(seed) # Makes the script deterministic

    # Generate the points
    F =  [ [random.gauss(0.,10.) for nn in range(n)] for mm in range(m)]
    f =  [random.gauss(0.,10.) for mm in range(m)]

    return (F,f)
   
