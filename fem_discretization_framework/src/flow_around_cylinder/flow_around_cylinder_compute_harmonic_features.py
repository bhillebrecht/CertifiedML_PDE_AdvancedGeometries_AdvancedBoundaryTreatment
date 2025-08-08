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


# import external libraries
import sys
import os 
import logging
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "util"))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from dolfinx import plot, fem, io
import pyvista

from pyDOE import lhs

import compute_areas_and_boundaries as cab

# import local functionality
# bht 2025-03-19: removed since functionality not yet transferred
from mode_computations import compute_eigenmodes, compute_derivative_information, get_boundary_and_domain_vertices, evaluate_function
from xdmf_helper import load_xdmf

logging.getLogger().setLevel(logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler())

##########################################################################################
# Parametrization of Computation ad Loading of Data
##########################################################################################
# number of modes and removed 0 modes
n_modes = 25

# number of modes
order = 1

# load and storage path
path_to_mesh = os.path.join(os.path.dirname(__file__), "..","..", "input_data","flow_around_cylinder", "flow_around_cylinder_mesh_xoff_0.5_only_dirichlet.xdmf")
path_to_data= os.path.join(os.path.dirname(__file__), "..","..", "output_data", "flow_around_cylinder", "mode_data_")

# load mesh
logging.info('Load data from '+path_to_mesh) 
mesh, facet_tags = load_xdmf(path_to_mesh)



##########################################################################################
# Computation and Storage of Auxiliary Information on the Domain
##########################################################################################
boundary_vertices, domain_vertices = get_boundary_and_domain_vertices(mesh, xlim=[0.0, 1.2])
coordinates = mesh.geometry.x

coordinates = coordinates[np.squeeze(np.argwhere(coordinates[:,0] >= 0.0)), :]
coordinates = coordinates[np.squeeze(np.argwhere(coordinates[:,0] <= 1.2)), :]

##########################################################################################
# Computation and Storage of Eigenvalues and Eigenmodes of the Laplace Beltrami
##########################################################################################
# bht 2025-03-19: removed since functionality not yet transferred
n_zero_modes = 0
n_modes_real = 0
while n_modes_real < n_modes:
    logging.info('Compute eigenmodes') 
    n_modes_w_offset = n_modes + n_zero_modes
    eigenvalues, eigenmodes, u_eigenmodes = compute_eigenmodes(mesh, n_modes=n_modes_w_offset, trace_frequency=10, is_pure_DC=True, order=order, mode=2, xlim=[0.0, 1.2])

    n_zero_modes = np.sum(np.sum(np.abs(eigenmodes[domain_vertices, :]), axis=0)<1e-6)
    n_modes_real = np.sum(np.sum(np.abs(eigenmodes[domain_vertices, :]), axis=0)>=1e-6)

    print(n_modes_real)


modes_dx, u_eigenmodes_dx = compute_derivative_information(u_eigenmodes, mesh, n_max = eigenmodes.shape[0], order = order, xlim=[0.0, 1.2])
modes_dxdx, u_eigenmodes_dxx = compute_derivative_information(u_eigenmodes_dx, mesh, n_max = eigenmodes.shape[0], order = order, xlim=[0.0, 1.2])

modes_dy, u_eigenmodes_dy = compute_derivative_information(u_eigenmodes, mesh, n_max = eigenmodes.shape[0], direction=1, order = order, xlim=[0.0, 1.2])
modes_dydy, u_eigenmodes_dyy = compute_derivative_information(u_eigenmodes_dy, mesh, n_max = eigenmodes.shape[0], direction=1, order = order, xlim=[0.0, 1.2])

# evaluate on additional points
n_influx = 400
points_influx = np.concatenate([np.zeros((n_influx, 1)), np.reshape(np.random.uniform(low=0.0, high=0.41, size=n_influx), (n_influx, 1)), np.zeros((n_influx, 1))], axis=1)
eigenmodes_influx, points_influx = evaluate_function(points_influx, u_eigenmodes, mesh, order = order)

n_walls = 400
points_walls = np.concatenate([np.reshape(np.random.uniform(low=0.0, high=1.2, size=n_influx), (n_influx, 1)), np.zeros((n_influx, 2))], axis=1)
eigenmodes_walls, points_walls = evaluate_function(points_walls, u_eigenmodes, mesh, order = order)

n_outlet = 400
points_outlet = np.concatenate([1.2*np.ones((n_outlet, 1)), np.reshape(np.random.uniform(low=0.0, high=0.41, size=n_outlet), (n_outlet, 1)), np.zeros((n_outlet, 1))], axis=1)
eigenmodes_outlet, _ = evaluate_function(points_outlet, u_eigenmodes, mesh, order = order)
eigenmodes_outlet_dx, _ = evaluate_function(points_outlet, u_eigenmodes_dx, mesh, order = order)

n_interior = 10000
ub = np.array([1.2, 0.41])
points_interior         = np.concatenate([ub*lhs(2, n_interior), np.zeros((n_interior, 1))], axis = 1)
eigenmodes_interior     , points_interior = evaluate_function(points_interior, u_eigenmodes, mesh, order = order)
eigenmodes_interior_dx  , points_interior = evaluate_function(points_interior, u_eigenmodes_dx, mesh, order = order)
eigenmodes_interior_dy  , points_interior = evaluate_function(points_interior, u_eigenmodes_dy, mesh, order = order)
eigenmodes_interior_dxx , points_interior = evaluate_function(points_interior, u_eigenmodes_dxx, mesh, order = order)
eigenmodes_interior_dyy , points_interior = evaluate_function(points_interior, u_eigenmodes_dyy, mesh, order = order)

# modes, modes_dx, modes_dy, modes_dxdx, modes_dydy = compute_mode_and_derivative_information(eigenmodes, mesh, n_max=n_modes_w_offset)

n_min = n_zero_modes
n_max = n_zero_modes+n_modes
np.save(path_to_data + "modes_influx.npy", np.concatenate([points_influx, eigenmodes_influx[:, n_min:n_max]], axis=1))
np.save(path_to_data + "modes_walls.npy", np.concatenate([points_walls, eigenmodes_walls[:, n_min:n_max]], axis=1))
np.save(path_to_data + "modes_outlet.npy", np.concatenate([points_outlet, eigenmodes_outlet[:, n_min:n_max], eigenmodes_outlet_dx[:, n_min:n_max]], axis=1))

np.save(path_to_data + "modes.npy", np.concatenate([eigenmodes[:, n_min:n_max],     eigenmodes_interior    [:, n_min:n_max]], axis=0) )
np.save(path_to_data + "modes_dx.npy", np.concatenate([modes_dx[:, n_min:n_max],    eigenmodes_interior_dx [:, n_min:n_max]], axis=0))
np.save(path_to_data + "modes_dy.npy", np.concatenate([modes_dy[:, n_min:n_max],    eigenmodes_interior_dy [:, n_min:n_max]], axis=0))
np.save(path_to_data + "modes_dxdx.npy", np.concatenate([modes_dxdx[:, n_min:n_max],eigenmodes_interior_dxx[:, n_min:n_max]], axis=0) )
np.save(path_to_data + "modes_dydy.npy", np.concatenate([modes_dydy[:, n_min:n_max],eigenmodes_interior_dyy[:, n_min:n_max]], axis=0) )


##########################################################################################
# Storage of Auxiliary Information on the Domain
##########################################################################################
np.save( path_to_data + "coordinates.npy", np.concatenate([coordinates, points_interior], axis=0))
# bht 2025-03-19: removed since functionality not yet transferred
np.save( path_to_data + "boundary_vertices.npy", boundary_vertices)
np.save( path_to_data + "domain_vertices.npy", np.concatenate( [domain_vertices, np.arange(eigenmodes.shape[0], eigenmodes.shape[0]+eigenmodes_interior.shape[0])], axis=0))


##########################################################################################
# Prep and store evaluation info
##########################################################################################
foldername = os.path.join(os.path.dirname(__file__), "..","..", "input_data","flow_around_cylinder")
mesh_vis, facet_tags = load_xdmf(os.path.join(foldername, "mesh_for_visualization.xdmf"))

# load evaluation coordinates 
coords_eval = np.load(os.path.join(foldername, "coordinates_for_visualization.npy"))
n_coords = coords_eval.shape[0]

# load reference result
df_reference = pd.read_csv(os.path.join(foldername, "reference_solution.csv"))
df_reference.drop(columns = ["x", "y"])

# compute boundary nodes, associated boundary lengths, and areas associated per node
dirichlet, nuemann, is_boundary, boundary_marking = cab.mark_boundary_nodes(mesh_vis)
is_boundary_facet = cab.mark_boundary_facets(mesh_vis, is_boundary)
boundary_len = cab.get_boundary_length(mesh_vis, is_boundary_facet)
_, _, iota_all = cab.get_iotas(mesh_vis, dirichlet, nuemann)
surface_boundary = cab.get_facet_area_boundary(mesh_vis, iota_all, is_boundary_facet, boundary_len)
volume_all = cab.get_area_all_cells_per_node(mesh_vis)
surface_all = np.zeros(volume_all.shape)
surface_all[iota_all] = surface_boundary

# evaluate eigenmodes
eigenmodes_eval     , _  = evaluate_function(coords_eval, u_eigenmodes, mesh, order = order)
eigenmodes_eval_dx  , _  = evaluate_function(coords_eval, u_eigenmodes_dx, mesh, order = order)
eigenmodes_eval_dy  , _  = evaluate_function(coords_eval, u_eigenmodes_dy, mesh, order = order)
eigenmodes_eval_dxx , _  = evaluate_function(coords_eval, u_eigenmodes_dxx, mesh, order = order)
eigenmodes_eval_dyy , _  = evaluate_function(coords_eval, u_eigenmodes_dyy, mesh, order = order)

df = pd.DataFrame({
    "x" : coords_eval[:,0],
    "y" : coords_eval[:,1],
    "boundary" : boundary_marking, 
    "boundary_length": surface_all,
    "area_per_node" : volume_all
})

if not os.path.exists(os.path.join(foldername, "reference_solution")):
    os.mkdir(os.path.join(foldername, "reference_solution"))

for i in range(n_min, n_max):
    df_tmp = pd.DataFrame({  "modes_"+str(i-n_min) : eigenmodes_eval[:, i]})
    df = pd.concat([df, df_tmp], axis=1)

for i in range(n_min, n_max):
    df_tmp = pd.DataFrame({  "modes_dx_"+str(i-n_min) : eigenmodes_eval_dx[:, i]})
    df = pd.concat([df, df_tmp], axis=1)

for i in range(n_min, n_max):
    df_tmp = pd.DataFrame({  "modes_dy_"+str(i-n_min) : eigenmodes_eval_dy[:, i]})
    df = pd.concat([df, df_tmp], axis=1)

for i in range(n_min, n_max):
    df_tmp = pd.DataFrame({  "modes_dxx_"+str(i-n_min) : eigenmodes_eval_dxx[:, i]})
    df = pd.concat([df, df_tmp], axis=1)

for i in range(n_min, n_max):
    df_tmp = pd.DataFrame({  "modes_dyy_"+str(i-n_min) : eigenmodes_eval_dyy[:, i]})
    df = pd.concat([df, df_tmp], axis=1)

for i in range(41):
    print("Write "+str(i)+"th file.")
    t = i*0.1
    df_temp = pd.DataFrame({
        "t" : np.ones(n_coords)*t,
        "u1" : df_reference["u1_sol_t"+'{0:.1f}'.format(t)],
        "u2" : df_reference["u2_sol_t"+'{0:.1f}'.format(t)],
        "p" :  df_reference["p_sol_t"+'{0:.1f}'.format(t)]
    })
    df_t = pd.concat([df, df_temp], axis=1)
    df_t.to_csv(os.path.join(foldername, "reference_solution", "input_for_evaluation_t_"+'{0:.1f}'.format(t)+".csv"))