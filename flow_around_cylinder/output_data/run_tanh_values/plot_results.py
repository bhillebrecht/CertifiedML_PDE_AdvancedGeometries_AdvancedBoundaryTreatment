# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: new_fenics
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Script to plot results based on FEM meshes for the Stokes flow around the cylinder
#
# ###################################################################################################
#  Copyright (c) 2025 Birgit Hillebrecht
#
#  To cite this code in publications, please use
#        B. Hillebrecht and B. Unger: "Prediction error certification for PINNs:
#        Theory, computation, and application to Stokes flow", arXiv preprint 
#        available.
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#  
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
#
# ###################################################################################################

# %%
import sys
import os
sys.path.append(os.path.join("..", "..", ".."))
sys.path.append(os.path.join(os.getcwd(), "..","..", "..", "fem_discretization_framework","src", "util"))

import numpy as np
import pandas as pd
from pyDOE import lhs
import pyvista
from dolfinx.plot import vtk_mesh
from dolfinx.io import XDMFFile
from dolfinx.fem import functionspace
from dolfinx import fem
from basix.ufl import element

from xdmf_helper import load_xdmf

# %%
clim_max = np.array([0.5, 1.0, 1.5, 2.0])

# %%
input_coordinates_file = os.path.join(os.getcwd(), "..", "..", "..", "fem_discretization_framework", "input_data", "flow_around_cylinder", "coordinates_for_visualization.npy")
coords_input = np.load(input_coordinates_file)

input_mesh_file = os.path.join(os.getcwd(), "..", "..", "..", "fem_discretization_framework", "input_data", "flow_around_cylinder", "mesh_for_visualization.xdmf")
mesh, facet_tags = load_xdmf(input_mesh_file)

# %%
remaining_indices = np.arange(1, mesh.geometry.x.shape[0])
A = np.zeros([mesh.geometry.x.shape[0], mesh.geometry.x.shape[0]])

for i in range(mesh.geometry.x.shape[0]):
    for j in range(remaining_indices.shape[0]):
        if np.sum((mesh.geometry.x[i, :] - coords_input[remaining_indices[j], :])**2) < 1e-6:
            A[i, remaining_indices[j]] = 1.0
            remaining_indices = np.delete(remaining_indices, j)
            break

# %%
dt = 0.1
clim_max_index = 0

# %%
for i in range(41):
    t = dt*i

    result_file = os.path.join(os.getcwd(), "run_tanh_input_for_evaluation_t_"+'{0:.1f}'.format(t)+".csv")
    if os.path.isfile(result_file):

        df = pd.read_csv(result_file)
        u1 = np.matmul(A,df["y0"].to_numpy())
        u2 = np.matmul(A, df["y1"].to_numpy())
        
        reference_file = os.path.join(os.getcwd(), "..", "..", "..", "fem_discretization_framework", "input_data", "flow_around_cylinder", "reference_solution", "input_for_evaluation_t_"+'{0:.1f}'.format(t)+".csv")
        df = pd.read_csv(reference_file)
        u1_ref = np.matmul(A, df["u1"].to_numpy())
        u2_ref = np.matmul(A, df["u2"].to_numpy())
        
        v_cg2 = element("Lagrange", mesh.topology.cell_name(), 1, shape=(mesh.geometry.dim, ))
        V = functionspace(mesh, v_cg2)
        
        topology, cell_types, geometry = vtk_mesh(V)

        while clim_max[clim_max_index] < np.max(np.sqrt(u1**2 + u2**2)) :
            clim_max_index +=1
        clim=[0, clim_max[clim_max_index]]
        
        # Create a pyvista-grid for the mesh
        mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim)
        grid = pyvista.UnstructuredGrid(*vtk_mesh(mesh, mesh.topology.dim))
        grid2 = pyvista.UnstructuredGrid(*vtk_mesh(mesh, mesh.topology.dim))
        
        grid["magnitude"] = np.sqrt(u1**2 + u2**2)
        grid2["ref"] = np.sqrt(u1_ref**2 + u2_ref**2) 
        
        # Create plotter
        plotter = pyvista.Plotter(shape=(3, 1), row_weights=[1.0, 1.0, 0.2], border=False)
        plotter.subplot(0, 0)
        plotter.add_mesh(grid, scalars="magnitude", cmap="viridis",clim=(clim[0], clim[1]), point_size=10, show_scalar_bar=False)
        #plotter.add_mesh(grid, style="wireframe", color="k")
        plotter.view_xy()
        plotter.zoom_camera(3.1)
        
        plotter.subplot(1, 0)
        plotter.add_mesh(grid2, scalars="ref", cmap="viridis",clim=(clim[0], clim[1]), point_size=10, show_scalar_bar=False)
        #plotter.add_mesh(grid2, style="wireframe", color="k")
        plotter.view_xy()
        plotter.zoom_camera(3.1)
        
        plotter.subplot(2, 0)
        polydata = pyvista.PolyData(np.zeros((2, 3)))
        polydata['data'] = (clim[0], clim[1])
        
        actor = plotter.add_mesh(polydata, scalars=None, show_scalar_bar=False)
        actor.visibility = False
        
        scalar_bar_kwargs = {
            'color': 'k',
            'title': actor.mapper.lookup_table._lookup_type + '\n',
            'outline': False,
            'title_font_size': 40,
        }
        label_level = 0
        if actor.mapper.lookup_table.below_range_color:
            scalar_bar_kwargs['below_label'] = 'below'
            label_level = 1
        if actor.mapper.lookup_table.above_range_color:
            scalar_bar_kwargs['above_label'] = 'above'
            label_level = 1
        
        label_level += actor.mapper.lookup_table._nan_color_set
        scalar_bar = plotter.add_scalar_bar(**scalar_bar_kwargs)
        scalar_bar.SetLookupTable(actor.mapper.lookup_table)
        scalar_bar.SetMaximumNumberOfColors(actor.mapper.lookup_table.n_values)
        scalar_bar.SetPosition(0.03, 0.1 + label_level * 0.1)
        scalar_bar.SetPosition2(0.95, 0.9 - label_level * 0.1)
        
        plotter.show()
        x = plotter.screenshot(os.path.join(os.getcwd(), "..", "run_tanh_figures","heatmap_comparison_t_"+'{0:.1f}'.format(t)+".png"))
        

# %%
