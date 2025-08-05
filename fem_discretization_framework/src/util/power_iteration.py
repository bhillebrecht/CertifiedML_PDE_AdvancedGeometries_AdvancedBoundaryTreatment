###################################################################################################
# Copyright (c) 2025 Birgit Hillebrecht
#
# To cite this code in publications, please use
#       B. Hillebrecht and B. Unger: "Prediction error certification for PINNs:
#       Theory, computation, and application to Stokes flow", arXiv preprint 
#       available.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
###################################################################################################

import scipy
import scipy.linalg 
import numpy as np

def power_iteration(A, startvector, tol, maxiter, printlog=False):
    mu = np.inf
    deltamu = tol*1000
    itercount = 0
    newvector = startvector
    newvector2 = A*newvector

    while deltamu > tol and itercount < maxiter :
        newvector = newvector2 / np.sqrt(np.dot(newvector2, newvector2))
        newvector2 = A*newvector
        newmu = np.dot(newvector, newvector2) 

        if printlog :
            print("Iteration: "+ str(itercount) + "    largest EV: " + str(newmu))

        deltamu = np.abs(mu - newmu)
        mu = newmu 
        itercount += 1
        
    return mu 