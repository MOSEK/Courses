import numpy
import math

import maxlikeden

# Testing using the exponentional distribution
n     = 10
y     = numpy.random.exponential(scale=1.0, size=n)
y     = numpy.sort(y)

xstar = maxlikeden.buildandsolve(y)

viol  = 0.0
a     = 0.0
for i in range(n-1):
    a = a+(y[i+1]-y[i])*(xstar[i]+xstar[i+1])

a = 0.5*a    
for i in range(n-2):
    viol   = max(viol,(xstar[i+1]-xstar[i])/(y[i+1]-y[i])-(xstar[i+2]-xstar[i+1])/(y[i+2]-y[i+1]))
    
print(y)
print(xstar)    

print('Area: %e Viol: %e min(x): %e\n' % (a,viol,numpy.min(xstar)))






