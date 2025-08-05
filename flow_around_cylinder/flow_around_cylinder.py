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

import os
import sys
import tensorflow as tf
import numpy as np
import pandas as pd
from pyDOE import lhs
import logging
import matplotlib.pyplot as plt
logger = logging.getLogger(__name__)

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "base_framework"))
sys.path.append(os.path.dirname(__file__))
from base.pinn import PINN
from helpers.csv_helpers import import_csv
from helpers.norms import set_current_norm
from weighted_l2_norms_for_flow import load_weights, weighted_l2_for_2d_FEM, weighted_l2_for_2d_FEM_from_contribs, weighted_l2_for_2d_FEM_single_point_contrib

###############################################################################
## FlowAroundCylinder
###############################################################################
class FlowAroundCylinder(PINN):

    def __init__(self, layers, lb, ub, X_f=None, problem_specifics=None):
        """
        Constructor.

        :param list layers: widths of the layers
        :param np.ndarray lb: lower bound of the inputs of the training data
        :param np.ndarray ub: upper bound of the inputs of the training data
        :param np.ndarray X_f: collocation points
        """

        super().__init__(layers, lb, ub)
        self.has_space_restriction = True
        self.problem_specifics = problem_specifics
        if self.problem_specifics != None and 'influx_weight' in self.problem_specifics:
            self.space_weight = [self.problem_specifics['influx_weight'], 
                                 self.problem_specifics['wall_weight'], 
                                 self.problem_specifics['outlet_weight'], 
                                 self.problem_specifics['div_free_weight']]    

        self.n_modes = 0
        self.path_to_files = None
        if self.problem_specifics != None and 'checkpoints_dir' in self.problem_specifics:
            self.checkpoints_dir = os.path.join(os.path.dirname(__file__), self.problem_specifics['checkpoints_dir'])
            if not os.path.isdir(self.checkpoints_dir):
                os.mkdir(self.checkpoints_dir)
        else:
            self.checkpoints_dir = os.path.join(os.path.dirname(__file__), "checkpoints")

        # internal restrictions
        self.t = None 
        self.phi = None # here, the number of inputs is directly related to the number of eigenmodes which were considered
        self.phi_dx = None # derivative of eigenmode-encoded coordinate w.r.t. x
        self.phi_dy = None # derivative of eigenmode-encoded coordinate w.r.t. y
        self.phi_dxx = None # second derivative of eigenmode-encoded coordinate w.r.t. x
        self.phi_dyy = None # second derivative of eigenmode-encoded coordinate w.r.t. y

        # soft boundary constraints
        self.t_space = None 
        self.x_space = None # not used
        self.y_space = None
        self.phi_space = None 

        # wall weight
        self.t_wall = None 
        self.phi_wall = None 

        # wall weight
        self.t_outlet = None 
        self.phi_outlet = None 
        self.phi_dx_outlet = None 

        # static parameters of the problem
        self.rho = 1.0
        self.mu = 1e-3
        self.Re = 100.0

        # load path information for input and set collocation points
        if self.problem_specifics != None and 'mesh' in self.problem_specifics:
            self.load_mesh_and_mode_data(os.path.join(os.path.dirname(__file__), self.problem_specifics['mesh']))
            self.t = self.tensor(lb[0] + (ub[0] - lb[0]) * lhs(1, self.phi.shape[0]))
            self.load_space_collocation()
        elif X_f is not None:
            self.set_collocation_points(X_f)

        self.has_D0 = False

    def load_mesh_and_mode_data(self, path):
        logger.info("Load mesh and mode data from: " + path)
        self.path_to_files = path
        inner_indices = np.load(os.path.join(path, "mode_data_domain_vertices.npy"))
        self.phi =  self.tensor(np.load(os.path.join(path, "mode_data_modes.npy"))[inner_indices, :])
        if os.path.isfile(os.path.join(path, "mode_data_modes_dx.npy")):
            self.phi_dx = self.tensor(np.load(os.path.join(path, "mode_data_modes_dx.npy"))[inner_indices, :])
            self.phi_dy =  self.tensor(np.load(os.path.join(path, "mode_data_modes_dy.npy"))[inner_indices, :])
            self.phi_dxx =  self.tensor(np.load(os.path.join(path, "mode_data_modes_dxdx.npy"))[inner_indices, :])
            self.phi_dyy =  self.tensor(np.load(os.path.join(path, "mode_data_modes_dydy.npy"))[inner_indices, :])

    def set_collocation_points(self, X_f):
        """
        not used because all collocation points are set in load_mesh_and_mode_data
        """
        return

    def load_space_collocation(self):
        # load relevant data
        logger.info("Load space collocation points from "+self.path_to_files+ " and set T_max to "+str(float(self.problem_specifics['Tmax'])))
        
        # define input and helper variables
        # influx data
        influx_data = np.load(os.path.join(self.path_to_files, "mode_data_modes_influx.npy"))
        self.phi_space =  self.tensor(influx_data[:, 3:influx_data.shape[1]])
        self.t_space =  self.tensor(float(self.problem_specifics['Tmax'])*lhs(1, self.phi_space.shape[0]))
        self.y_space = self.tensor(influx_data[:, 1:2])

        n_modes = influx_data.shape[1]-3

        # wall data
        wall_data = np.load(os.path.join(self.path_to_files, "mode_data_modes_walls.npy"))
        self.phi_wall =  self.tensor(wall_data[:, 3:wall_data.shape[1]])
        self.t_wall =  self.tensor(float(self.problem_specifics['Tmax'])*lhs(1, self.phi_wall.shape[0]))

        # outlet data
        outlet_data = np.load(os.path.join(self.path_to_files, "mode_data_modes_outlet.npy"))
        self.phi_outlet =  self.tensor(outlet_data[:, 3:3+n_modes])
        self.phi_outlet_dx =  self.tensor(outlet_data[:, 3+n_modes:3+2*n_modes])
        self.t_outlet =  self.tensor(float(self.problem_specifics['Tmax'])*lhs(1, self.phi_outlet.shape[0]))

        # count all boundary losses
        self.n_boundary = self.phi_outlet.shape[0]+self.phi_space.shape[0]+self.phi_wall.shape[0]

    def set_space_collocation(self, X_space):
        """
        This function sets the collocation points on the boundary 
        """
        logging.info("Space collocation")

        if X_space == None:
            self.load_space_collocation()
        else:
            logger.info("Set space collocation according to X_space")
            self.t_space = self.tensor(X_space[:,0:1]) # embedded coordinates in eigenfunctions + one temporal coordinate
            self.phi_space = self.tensor(X_space[:,1:self.n_modes+1]) # embedded coordinates in eigenfunctions + one temporal coordinate

    @tf.function
    def f_model(self, X_f=None):
        """
        The actual PINN to approximate the motion of the considered ODE/PDE

        :return: tf.Tensor: the prediction of the PINN
        """

        if X_f is None:
            t = self.t
            phi = self.phi
            phi_dx = self.phi_dx
            phi_dy = self.phi_dy
            phi_dxx = self.phi_dxx
            phi_dyy = self.phi_dyy
        else:
            t = self.tensor(X_f[:,0:1]) # embedded coordinates in eigenfunctions + one temporal coordinate
            phi = self.tensor(X_f[ : , 1 : self.n_modes+1]) # embedded coordinates in eigenfunctions + one temporal coordinate
            phi_dx = self.tensor(X_f[ : , self.n_modes + 1 : 2 * self.n_modes])
            phi_dy = self.tensor(X_f[ : , 2 * self.n_modes + 1 : 3 * self.n_modes])
            phi_dxx = self.tensor(X_f[ : , 3 * self.n_modes + 1 : 4 * self.n_modes])
            phi_dyy = self.tensor(X_f[ : , 5 * self.n_modes + 1 : 6 *self.n_modes])


        with tf.GradientTape(persistent = True) as tapephi:
            tapephi.watch(phi)
            with tf.GradientTape(persistent = True) as tapephiphi, tf.GradientTape(persistent = True) as tapet:
                tapephiphi.watch(phi)
                tapet.watch(t)
                y = self.model(tf.concat([t, phi], axis = 1))

                v1 = y[:, 0:1]
                v2 = y[:, 1:2]
                p  = y[:, 2:3]

            dv_dt = tf.concat([tapet.gradient(v1, t), tapet.gradient(v2, t)], axis = 1)
            
            # cauchy stress tensor σ=−pI+2μD(v)
            # D(v) = 1/2*(jac(v) + jac(v)^T)
  
            gradv1 = tapephiphi.gradient(v1, phi)
            gradv2 = tapephiphi.gradient(v2, phi)
            gradp = tapephiphi.gradient(p, phi)

            dv1_dx = tf.reduce_sum(tf.multiply(gradv1, phi_dx), axis=1, keepdims = True)
            dv1_dy = tf.reduce_sum(tf.multiply(gradv1, phi_dy), axis=1, keepdims = True)
            dv2_dx = tf.reduce_sum(tf.multiply(gradv2, phi_dx), axis=1, keepdims = True)
            dv2_dy = tf.reduce_sum(tf.multiply(gradv2, phi_dy), axis=1, keepdims = True)
            dp_dx  = tf.reduce_sum(tf.multiply(gradp, phi_dx), axis=1, keepdims = True)
            dp_dy  = tf.reduce_sum(tf.multiply(gradp, phi_dy), axis=1, keepdims = True)

        # second derivatives of v_x
        dv1_dxx =  tf.reduce_sum(tf.multiply(tapephi.gradient(dv1_dx, phi), phi_dx), axis=1, keepdims = True) 
        dv1_dyy =  tf.reduce_sum(tf.multiply(tapephi.gradient(dv1_dy, phi), phi_dy), axis=1, keepdims = True) 
        dv1_dxx += tf.reduce_sum(tf.multiply(gradv1, phi_dxx), axis=1, keepdims = True) 
        dv1_dyy += tf.reduce_sum(tf.multiply(gradv1, phi_dyy), axis=1, keepdims = True) 

        # second derivatives of v_y
        dv2_dxx =  tf.reduce_sum(tf.multiply(tapephi.gradient(dv2_dx, phi), phi_dx), axis=1, keepdims = True) 
        dv2_dyy =  tf.reduce_sum(tf.multiply(tapephi.gradient(dv2_dy, phi), phi_dy), axis=1, keepdims = True) 
        dv2_dxx += tf.reduce_sum(tf.multiply(gradv2, phi_dxx), axis=1, keepdims = True) 
        dv2_dyy += tf.reduce_sum(tf.multiply(gradv2, phi_dyy), axis=1, keepdims = True) 

        grad_sigma = tf.concat([dp_dx - self.mu*(dv1_dxx + dv1_dyy ),  dp_dy - self.mu*(dv2_dxx + dv2_dyy )], axis=1)
        #nonlinear_1 = tf.concat([tf.multiply(v1, dv1_dx) + tf.multiply(v2, dv1_dy), tf.multiply(v1, dv2_dx) + tf.multiply(v2, dv2_dy)], axis = 1)

        f_pred = self.rho*dv_dt + grad_sigma #+ self.rho * nonlinear_1

        return f_pred
    
    def use_input_data_for_space_collocation(self, filepath):
        if filepath[-3:] == "csv":
            df = pd.read_csv(filepath)
            n_coords = df["x"].shape[0]
            X_data = np.zeros([n_coords, 1+5*25+1])
            X_data[:, 0] = df["t"]
            counter = 1
            for i in range(25):
                X_data[:, counter] = df["modes_"+str(i)]
                counter += 1
            for i in range(25):
                X_data[:, counter] = df["modes_dx_"+str(i)]
                counter += 1
            for i in range(25):
                X_data[:, counter] = df["modes_dy_"+str(i)]
                counter += 1
            for i in range(25):
                X_data[:, counter] = df["modes_dxx_"+str(i)]
                counter += 1
            for i in range(25):
                X_data[:, counter] = df["modes_dyy_"+str(i)]
                counter += 1
            X_data[:, counter] = df["boundary"]
            Y_data = np.zeros([n_coords, 3])
            Y_data[:, 0]

        inlet = 1
        outlet = 2
        wall = 3

        boundary_markings = df["boundary"].to_numpy(dtype=np.float64)
        inlet_indices = np.argwhere(boundary_markings == inlet)
        outlet_indices = np.argwhere(boundary_markings == outlet)
        wall_indices = np.argwhere(boundary_markings == wall)
        
        boundary_markings_compressed = boundary_markings[np.nonzero(boundary_markings)]
        self.n_boundary = boundary_markings_compressed.shape[0]

        self.inlet_indices_compressed = np.squeeze(np.argwhere(boundary_markings_compressed == inlet))
        self.outlet_indices_compressed = np.squeeze(np.argwhere(boundary_markings_compressed == outlet))
        self.wall_indices_compressed = np.squeeze(np.argwhere(boundary_markings_compressed == wall))

        ## inlet
        self.t_space = self.tensor(df["t"].to_numpy(dtype=np.float64)[inlet_indices])
        self.phi_space = np.zeros([self.t_space.shape[0], self.input_dim-1])
        for i in range(25):
            self.phi_space[:, i:i+1] = df["modes_"+str(i)].to_numpy(dtype=np.float64)[inlet_indices]
        self.phi_space = self.tensor(self.phi_space)
        self.y_space = self.tensor(df["y"].to_numpy(dtype=np.float64)[inlet_indices])

        ## walls 
        self.t_wall = self.tensor(df["t"].to_numpy(dtype=np.float64)[wall_indices])
        self.phi_wall = np.zeros([self.t_wall.shape[0], self.input_dim-1])
        for i in range(25):
            self.phi_wall[:, i:i+1] = df["modes_"+str(i)].to_numpy(dtype=np.float64)[wall_indices]
        self.phi_wall = self.tensor(self.phi_wall)

        ### outlet
        self.t_outlet = self.tensor(df["t"].to_numpy(dtype=np.float64)[outlet_indices])
        self.phi_outlet = np.zeros([self.t_outlet.shape[0], self.input_dim-1])
        self.phi_outlet_dx = np.zeros([self.t_outlet.shape[0], self.input_dim-1])
        for i in range(25):
            self.phi_outlet[:, i:i+1] = df["modes_"+str(i)].to_numpy(dtype=np.float64)[outlet_indices]
            self.phi_outlet_dx[:, i:i+1] = df["modes_dx_"+str(i)].to_numpy(dtype=np.float64)[outlet_indices]
        self.phi_outlet = self.tensor(self.phi_outlet)
        self.phi_outlet_dx = self.tensor(self.phi_outlet_dx)

    def space_model_inlet(self, temporal_derivative=False, replace_time = False, time = 0.0):
        ####################################################################################
        # input at x = 0
        ####################################################################################
        # expected
        if replace_time:
            t = tf.ones(self.t_space.shape, dtype=tf.float64)*time
        else:
            t = self.t_space
    
        H = 0.41 
        u = 1.5* tf.math.sin(t*np.pi/8)
        v_x = 4.0/(H*H)*u*self.y_space*(H-self.y_space)

        # reality
        with tf.GradientTape(persistent = True) as tapet:
            tapet.watch(t)
            q = self.model(tf.concat((t, self.phi_space), axis=1))

            v1 = q[:, 0:1]
            v2 = q[:, 1:2]
            p  = q[:, 2:3]
        dinletdt = tf.concat([tapet.gradient(v1, t), tapet.gradient(v2, t), tf.zeros([v_x.shape[0], 1], dtype='float64')], axis = 1)

        # concatenate along axis 0 to be able to concatenate with the divergence freeness term
        # pressure not considered
        inlet = tf.concat([v_x, tf.zeros([v_x.shape[0], 2], dtype='float64')], axis=1) - tf.concat([q[:, 0:2], tf.zeros([v_x.shape[0], 1], dtype='float64')], axis=1)

        return inlet, dinletdt
    
    def space_model_walls(self, temporal_derivative=False, replace_time = False, time = 0.0):
        ####################################################################################
        # walls
        ####################################################################################
        if replace_time:
            t = tf.ones(self.t_wall.shape, dtype=tf.float64)*time
        else:
            t = self.t_wall
    
        # reality
        with tf.GradientTape(persistent = True) as tapet:
            tapet.watch(t)
            q = self.model(tf.concat((t, self.phi_wall), axis=1))

            v1 = q[:, 0:1]
            v2 = q[:, 1:2]
            p  = q[:, 2:3]
        dwalldt = tf.concat([tapet.gradient(v1, t), tapet.gradient(v2, t), tapet.gradient(p, t)], axis = 1)

        # concatenate along axis 0 to be able to concatenate with the divergence freeness term
        # pressure not considered
        return q, dwalldt
    
    def space_model_outlet(self, temporal_derivative=False, replace_time = False, time = 0.0):
        ####################################################################################
        # walls
        ####################################################################################
        if replace_time:
            t = tf.ones(self.t_outlet.shape, dtype=tf.float64)*time
        else:
            t = self.t_outlet
        phi = self.phi_outlet
    
        # reality
        with tf.GradientTape(persistent = True) as tapet:
            tapet.watch(t)
            with tf.GradientTape(persistent=True) as tapephi:
                tapephi.watch(phi)
                q = self.model(tf.concat((t, phi), axis=1))

                v1 = q[:, 0:1]
                v2 = q[:, 1:2]
                p  = q[:, 2:3]

            gradv1 = tapephi.gradient(v1, phi)
            gradv2 = tapephi.gradient(v2, phi)
            dv1_dx = tf.reduce_sum(tf.multiply(gradv1, self.phi_outlet_dx), axis=1, keepdims = True)
            dv2_dx = tf.reduce_sum(tf.multiply(gradv2, self.phi_outlet_dx), axis=1, keepdims = True)

            outlet1 = dv1_dx-1/self.mu * p
            outlet2 = dv2_dx
            
        doutlet1dt = tapet.gradient(outlet1, t)
        doutlet2dt = tapet.gradient(outlet2, t)

        # concatenate along axis 0 to be able to concatenate with the divergence freeness term
        # pressure not considered
        return tf.concat([outlet1, outlet2, tf.zeros(outlet1.shape, dtype=np.float64)], axis = 1), tf.concat([doutlet1dt, doutlet2dt, tf.zeros(outlet1.shape, dtype=np.float64)], axis=1)
    
    def space_model(self, temporal_derivative=False, replace_time = False, time = 0.0):
        
        inlet, dinletdt = self.space_model_inlet(temporal_derivative, replace_time, time)
        wall, dwalldt = self.space_model_walls(temporal_derivative, replace_time, time)
        outlet, doutletdt = self.space_model_outlet(temporal_derivative, replace_time, time)

        if temporal_derivative and replace_time:
            inlet_error = tf.concat([inlet, dinletdt], axis=1)
            wall_error = tf.concat([wall, dwalldt], axis=1)
            outlet_error = tf.concat([outlet, doutletdt], axis=1)

            return_value = np.zeros([self.n_boundary, inlet_error.shape[1]], dtype = np.float64)
            return_value[self.inlet_indices_compressed, :] = inlet_error.numpy()
            return_value[self.wall_indices_compressed, :] = wall_error.numpy()
            return_value[self.outlet_indices_compressed, :] = outlet_error.numpy()
            return self.tensor(return_value)
        
        else:
            if temporal_derivative:
                inlet = tf.concat([inlet, dinletdt], axis=1)
                wall = tf.concat([wall, dwalldt], axis=1)
                outlet = tf.concat([outlet, doutletdt], axis=1)

            ####################################################################################
            # divergence freeness in velocity
            ####################################################################################
            t = self.t
            phi = self.phi
            phi_dx = self.phi_dx
            phi_dy = self.phi_dy
            with tf.GradientTape(persistent = True) as tapephi:
                tapephi.watch(phi)
                y = self.model(tf.concat([t, phi], axis = 1))
                v1 = y[:, 0:1]
                v2 = y[:, 1:2]

            gradv1 = tapephi.gradient(v1, phi)
            gradv2 = tapephi.gradient(v2, phi)

            dv1_dx = tf.reduce_sum(tf.multiply(gradv1, phi_dx), axis=1, keepdims = True)
            dv2_dy = tf.reduce_sum(tf.multiply(gradv2, phi_dy), axis=1, keepdims = True)

            div_free = dv1_dx + dv2_dy

            return inlet, wall, outlet, div_free
       
    def get_boundary_error_factors(self):
        """
        returns boundary factors as [norm(AB0), norm(B0)]
        """
        bf = np.array([0.001, 0.19])
        return bf

    def get_Lf(self):
        """
        returns lipschitz constant or spectral abscissa of right hand side of ODE
        """
        return 0.014

    def get_M(self):
        return 1.0

###############################################################################
## Callouts and Factory Functions
###############################################################################

def load_data(filepath):
    """
    Loads initial data
    """
    if filepath[-3:] == "npy":
        X_data = np.load(filepath)
        X_data = np.concatenate([np.zeros( (X_data.shape[0], 1)), X_data], axis=1)
        return X_data.shape[0], X_data, np.zeros( (X_data.shape[0], 3) )
    
    elif filepath[-3:] == "csv":
        df = pd.read_csv(filepath)
        n_coords = df["x"].shape[0]
        X_data = np.zeros([n_coords, 1+5*25+1])
        X_data[:, 0] = df["t"]
        counter = 1
        for i in range(25):
            X_data[:, counter] = df["modes_"+str(i)]
            counter += 1
        for i in range(25):
            X_data[:, counter] = df["modes_dx_"+str(i)]
            counter += 1
        for i in range(25):
            X_data[:, counter] = df["modes_dy_"+str(i)]
            counter += 1
        for i in range(25):
            X_data[:, counter] = df["modes_dxx_"+str(i)]
            counter += 1
        for i in range(25):
            X_data[:, counter] = df["modes_dyy_"+str(i)]
            counter += 1
        X_data[:, counter] = df["boundary"]
        Y_data = np.zeros([n_coords, 3])
        Y_data[:, 0]

        load_weights(filepath)
    
        return n_coords, X_data, Y_data

def create_pinn(nn_params, lb, ub, problem_specifics):
   """
   Factory function to create the target PINN 
   """
   pinn = FlowAroundCylinder(nn_params, lb, ub, problem_specifics=problem_specifics)

   set_current_norm(weighted_l2_for_2d_FEM, weighted_l2_for_2d_FEM_single_point_contrib, weighted_l2_for_2d_FEM_from_contribs)

   return pinn


def post_train_callout(pinn, output_directory) -> None:
    """
    Callout to be called after training. 

    :param PINN pinn: pinn of type (here) FlowAroundCylinder, which has been trained 
    :param string output_directory: path to output directory
    """
    u = pinn.model(tf.concat([tf.ones(pinn.t_space.shape, dtype=tf.float64)*3.0, pinn.phi_space], axis=1))
    plt.plot(pinn.y_space, u.numpy()[:,0], 'k.')
    plt.show()

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
    return