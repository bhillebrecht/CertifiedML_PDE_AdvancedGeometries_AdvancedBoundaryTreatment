###################################################################################################
#
# This file incorporates work and modifications to the originally published code
# according to the previous license by the following contributors under the following licenses
# 
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
# Copyright (c) 2022 Birgit Hillebrecht
#
# To cite this code in publications, please use
#       B. Hillebrecht and B. Unger : "Certified machine learning: Rigorous a posteriori error bounds for PDE defined PINNs", arxiV preprint available
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

import json
import os
import tensorflow as tf
import numpy as np

from base.pinn import PINN
from helpers.csv_helpers import export_csv, import_csv
from helpers.globals import get_prefix
from helpers.input_generator import generate_collocation_points
from helpers.integrate import compute_integral_trpz
from helpers.nn_parametrization import get_param_as_float, has_param
from helpers.plotting import new_fig, save_fig

###############################################################################
## PINN modelling the solution to the 1D heat equation
###############################################################################
class HeatEquation(PINN):

    def __init__(self, layers, lb, ub, X_f=None, bcs=None):
        """
        Constructor.

        :param list layers: widths of the layers
        :param np.ndarray lb: lower bound of the inputs of the training data
        :param np.ndarray ub: upper bound of the inputs of the training data
        :param np.ndarray X_f: collocation points
        """

        super().__init__(layers, lb, ub, bcs=bcs)
        self.t = None
        self.x = None

        self.t_space = None
        self.x_space = None

        self.checkpoints_dir = os.path.join(os.path.dirname(__file__), "output_data", "checkpoints")
        self.has_D0 = True 

        if X_f is not None:
            self.set_collocation_points(X_f)

    def set_collocation_points(self, X_f):
      self.t = self.tensor(X_f[:,0:1])
      self.x = self.tensor(X_f[:,1:2])

    def set_space_collocation(self, X_space):
        self.space_weight = 20.0*np.array([1.0, 1.0])     
        self.has_space_restriction = True

        self.x_space = X_space[:,1:2]
        self.t_space = X_space[:,0:1]

    def use_input_data_for_space_collocation(self, input_file):
        return

    @tf.function
    def f_model(self, X_f=None):
        """
        The actual PINN to approximate the motion of the heat equation

        :return: tf.Tensor: the prediction of the PINN
        """

        if X_f is None:
            t = self.t
            x = self.x
        else:
            t = self.tensor(X_f[:, 0:1])
            x = self.tensor(X_f[:, 1:2])

        with tf.GradientTape() as tapet:
            tapet.watch(t)
            with tf.GradientTape() as tapex:
                tapex.watch(x)
                with tf.GradientTape(persistent=True) as tapex2:
                    tapex2.watch(x)
                    q = self.model(tf.concat((t, x), axis=1), training=False)
                dq_dx = tapex2.gradient(q, x)
            dq_dxx = tapex.gradient(dq_dx, x)
        dq_dt = tapet.gradient(q, t)
        del tapex2
        f_pred = dq_dt-0.2*dq_dxx

        del tapex
        return f_pred

    def space_model(self, temporal_derivative = False, replace_time = False, time = 0.0):
        if replace_time:
            t = self.tensor(np.ones(self.t_space.shape)*time)
        else:
            t = self.tensor(self.t_space)
        
        if temporal_derivative:
            with tf.GradientTape() as tapet:
                tapet.watch(t)
                q = self.model(tf.concat((t, self.x_space), axis=1), training=False)
            dqdt = tapet.gradient(q, t)
            return tf.concat([q, dqdt], axis=1)
        else:
            q = self.model(tf.concat((t, self.x_space), axis=1), training=False)
            return q 

    def get_Lf(self):
        """
        returns spectral abscissa of right hand side of PDE
        """
        #return 0 # contractive
        return  (- 0.2*np.pi**2) # expdecay

    def get_M(self):
        return 1.0

    def get_boundary_error_factors(self):
        """
        Returns numpy array consisting of the norm of D0 and the norm of AD0
        """
        return np.array([2/3, 0.0])

    def has_hard_constraints(self):
        return False
    
    def evaluate_D0_delta0(self, X):
        boundary_values_at_t0 = self.model(self.tensor(np.array([[0.0, 0.0], [0.0, 1.0]])), training=False)
        return X*boundary_values_at_t0.numpy()[1] + (1.0 - X)*boundary_values_at_t0.numpy()[0]

###############################################################################
## Callouts and Factory Functions
###############################################################################

def load_data(filepath):
   """
   Loads data from csv file specific for the new application
   """
   data = import_csv(filepath)
   x_data = data[:, 0:2]
   y_data = None
   if data.shape[1]>2:
       y_data = data[:, 2:3]
   n_data = data.shape[0]

   return n_data, np.float64(x_data), np.float64(y_data)

def create_pinn(nn_params, lb, ub, problem_specifics = None):
   """
   Factory function to create the target PINN 
   """
   pinn = HeatEquation(nn_params, lb, ub)
   times = generate_collocation_points(100, lb[0:1], ub[0:1])
   times = np.append(times, 0)
   times = np.reshape(times, (times.shape[0], 1))
   dirichlet = np.concatenate( [np.concatenate([times, np.ones(times.shape)*lb[1]], axis=1), np.concatenate([times, np.ones(times.shape)*ub[1]], axis=1)], axis=0)
   pinn.set_space_collocation(dirichlet)

   return pinn

def post_train_callout(pinn, output_directory) -> None:
    """
    Callout to be called after training. 

    :param PINN pinn: pinn of type (here) TemplateAppl, which has been trained 
    :param string output_directory: path to output directory
    """
    export_csv(tf.concat([pinn.t, pinn.x], axis=1).numpy(), os.path.join(pinn.checkpoints_dir, "collocation_points.csv"))

    t = np.zeros((101,1))
    t2 = np.ones((101,1))*0.2
    t3 = np.ones((101,1))*0.1
    x = np.reshape(np.linspace(0,1,101), (101,1))

    input = np.concatenate((t, x), axis=1)
    input2 = np.concatenate((t2, x), axis=1)
    input3 = np.concatenate((t3, x), axis=1)

    y_pred, _ = pinn.predict(input)
    y_pred2, _ = pinn.predict(input2)
    y_pred3, _ = pinn.predict(input3)

    fig = new_fig()
    ax = fig.add_subplot(1, 1, 1)

    colors = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red']
    ax.set( ylabel=r'$u(t,x)$')
    ax.set( xlabel="$x$")

    ax.grid(True, which='both')
    ax.set(xlim=[0,1])
    ax.set(ylim=[-1.1,1.1])

    ax.plot(x, y_pred[:,0], linewidth=1.5, color=colors[0], linestyle="-", label="pred t=0")
    ax.plot(x, y_pred3[:,0], linewidth=1.5, color=colors[1], linestyle="-.", label="pred t=0.1")
    ax.plot(x, y_pred2[:,0], linewidth=1.5, color=colors[2], linestyle=":", label="pred t=0.2")
    ax.plot(x, np.sin(2*np.pi*x), linewidth=1.5, color=colors[3], linestyle=":", label="ref t=0")

    ax.legend(loc='best')
    fig.tight_layout()

    save_fig(fig, "output_training" , output_directory)

    return

def post_extract_callout(input, R, output_directory) -> None:
    """
    Callout to be called after extracting parameters necessary for a posteriori error estimation

    :param np.array input: input data on which the parameter estimation has been taken place
    :param np.array R: approximation error of the ODE/PDE for this input set
    :param np.array delta: smoothened approximation error of the ODE/PDE for this input set
    :param np.array deltadot: derivative of smoothened approximation error of the ODE/PDE for this input set
    :param np.array deltadotdot: second derivative of smoothened approximation error of the ODE/PDE for this input set
    :param string output_directory: path to output directory
    """
    
    return

def post_eval_callout(outdir, X_data, Y_pred, E_pred, N_SP_pred, Y_data=None, tail=None ) -> None:

    """
    Callout to be called after running NN for a testdataset X_data

    :param string outdir: path to output directory
    :param np.array X_data: input data set on which the NN has been evaluated
    :param np.array Y_pred: prediction of NN of system state
    :param np.array E_pred: predicted error (two components if numerical integration is used)
    :param np.array N_SP_pred: predicted number of supportpoints for numerical integration
    :param np.array Y_data: reference output data set (optional)

    """
#    fig = new_fig()
#    ax = fig.add_subplot(1, 1, 1)
#
#    colors = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red']
#    ax.set( ylabel=r'$u(t,x)$')
#    ax.set( xlabel="$x$")
#
#    ax.grid(True, which='both')
#    ax.set(xlim=[0,1])
#    ax.set(ylim=[-1.1,1.1])
#
#    ax.plot(X_data[:,1:2], Y_pred[:,:], linewidth=1.5, color=colors[0], linestyle="-", label="pred")
#    ax.plot(X_data[:,1:2], Y_data[:,:], linewidth=1.5,  color=colors[2], linestyle="-.", label="ref")
#
#    ax.legend(loc='best')
#    fig.tight_layout()
##    save_fig(fig, "output_"+ str(X_data[0,0]) , outdir)
    
    with open(os.path.join(outdir, "..", "run_"+get_prefix()+"values", "run_" + get_prefix() +tail[:-4] + "_error.json"), 'r') as file:
        data = file.read()
        data = json.loads(data)

    error =  (Y_pred[:,0] - Y_data[:,0])**2
    data['E_ref'] = np.sqrt(compute_integral_trpz(error, X_data[1,1]- X_data[0,1]))

    out_file = open( os.path.join(outdir, "..", "run_"+get_prefix()+"values", "run_" + get_prefix() +tail[:-4] + "_error.json"), "w")
    json.dump(data, out_file, indent=4)
    out_file.close()

    return

