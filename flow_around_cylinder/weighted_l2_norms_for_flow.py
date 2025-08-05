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

import logging
import sys
import numpy as np
import pandas as pd
import tensorflow as tf

WEIGHTS_AREA = None
WEIGHTS_EDGES = None

N_CELLS = 0
N_EDGES = 0

def load_weights(filepath):
    global WEIGHTS_EDGES, WEIGHTS_AREA, N_EDGES, N_CELLS

    df = pd.read_csv(filepath)
    n_coords = df["x"].shape[0]
    WEIGHTS_AREA = df["area_per_node"].to_numpy()
    N_CELLS = WEIGHTS_AREA.shape[0]

    WEIGHTS_EDGES = df["boundary_length"].to_numpy()
    WEIGHTS_EDGES = WEIGHTS_EDGES[np.nonzero(WEIGHTS_EDGES)[0]]
    N_EDGES = WEIGHTS_EDGES.shape[0]


def weighted_l2_for_2d_FEM_single_point_contrib(f, outdim= None):
    """
    Computes contribution of a single value to the L2 norm

    param np.array or tf.tensor f : array over which the L2 norm shall be computed
    """
    # assert correct parameter hand over
    if outdim is not None:
        if not isinstance(outdim, int):
            logging.error("Coding Error: the number of relevant output dimensions must be integer")
            sys.exit() 

    # reduce output dimensions if necessary
    if outdim is not None:
        f = f[:,:outdim]

    val = tf.square(f)

    # reduce dimensions if necessary
    if tf.rank(val) >1:
        val =  tf.reduce_sum( val , axis=1)

    return val

def weighted_l2_for_2d_FEM_from_contribs(single_point_contribs, x):
    """
    Computes L2 norm

    param np.array or tf.tensor f : array over which the L2 norm shall be computed
    param np.array x : underlying space grid 
    param int outdim : number of output dimesnions relevant for the L2 norm. Consider f[:, :outdim] for L2 norm only
    """
    global N_EDGES, N_CELLS, WEIGHTS_AREA, WEIGHTS_EDGES
    # convert tf to np if necessary
    if tf.is_tensor(single_point_contribs):
        single_point_contribs = single_point_contribs.numpy()

    if single_point_contribs.shape[0] == N_CELLS:
        I1 = np.sqrt(np.sum(single_point_contribs*WEIGHTS_AREA))
    elif single_point_contribs.shape[0] == N_EDGES:
        I1 = np.sqrt(np.sum(single_point_contribs*WEIGHTS_EDGES))
    else:
        logging.error("Number of contributions to integrate over does neither correspond to number of boundary edges nor to number of cells.")

    return np.sqrt(I1), 0.0, 0.0

def weighted_l2_for_2d_FEM(f, x, outdim=None):
    """
    Computes L2 norm

    param np.array or tf.tensor f : array over which the L2 norm shall be computed
    param np.array x : underlying space grid 
    param int outdim : number of output dimesnions relevant for the L2 norm. Consider f[:, :outdim] for L2 norm only
    """
    # sum over relevant output dimensions
    f = weighted_l2_for_2d_FEM_single_point_contrib(f, outdim)
    I1, I2, E = weighted_l2_for_2d_FEM_from_contribs(f, x)

    # return
    return I1, I2, E