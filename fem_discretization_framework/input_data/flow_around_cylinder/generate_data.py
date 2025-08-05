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
# # Script to generate a 2D mesh with a circular cutout for PINN simulation
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
#

# %%
import gmsh
import os
import numpy as np
import matplotlib.pyplot as plt

from mpi4py import MPI
from petsc4py import PETSc

from basix.ufl import element

from dolfinx.fem.petsc import (apply_lifting, assemble_matrix, assemble_vector,
                               create_vector, create_matrix, set_bc)
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
from dolfinx.io import (XDMFFile, distribute_entity_data, gmshio)
from dolfinx.mesh import create_mesh, meshtags_from_entities, locate_entities_boundary

gmsh.initialize()

# %%
# dimension mesh
gdim = 2

##### Rectangle
# x coordinates box
x_0_base = 0.0
L = 1.2

# y coordinates box
y_0 = 0.0
H = 0.41

# buffer to allow for non-trivial inlets and outlets
x_buffer = 0.0
x_0 = x_0_base - x_buffer

##### Circular obstacle
# position and dimension obstacle
c_x = c_y = 0.2
r = 0.05

##### BC modelling (drop inlet and outlet)
only_dirichlet = False

# %%
# initialize everything
mesh_comm = MPI.COMM_WORLD
model_rank = 0
if mesh_comm.rank == model_rank:
    rectangle = gmsh.model.occ.addRectangle(x_0, y_0, 0, L+2*x_buffer, H, tag=1)
    obstacle = gmsh.model.occ.addDisk(c_x, c_y, 0, r, r)

# %%
# cut the obstacle and synchronize the geometry
if mesh_comm.rank == model_rank:
    fluid = gmsh.model.occ.cut([(gdim, rectangle)], [(gdim, obstacle)])
    gmsh.model.occ.synchronize()

# %%
# mark volume elements
fluid_marker = 1
if mesh_comm.rank == model_rank:
    volumes = gmsh.model.getEntities(dim=gdim)
    assert (len(volumes) == 1)
    gmsh.model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], fluid_marker)
    gmsh.model.setPhysicalName(volumes[0][0], fluid_marker, "Fluid")

# %%
# mark walls according to boundary conditions
inlet_marker, outlet_marker, wall_marker, obstacle_marker = 2, 3, 4, 5
inflow, outflow, walls, obstacle = [], [], [], []
if mesh_comm.rank == model_rank:
    boundaries = gmsh.model.getBoundary(volumes, oriented=False)
    for boundary in boundaries:
        center_of_mass = gmsh.model.occ.getCenterOfMass(boundary[0], boundary[1])
        if np.allclose(center_of_mass, [x_0, H / 2, 0]):
            if not only_dirichlet:
                inflow.append(boundary[1])
        elif np.allclose(center_of_mass, [x_0+L+2*x_buffer, H / 2, 0]):
            if not only_dirichlet:
                outflow.append(boundary[1])
        elif np.allclose(center_of_mass, [L / 2, H, 0]) or np.allclose(center_of_mass, [L / 2, 0, 0]):
            walls.append(boundary[1])
        else:
            obstacle.append(boundary[1])
    gmsh.model.addPhysicalGroup(1, walls, wall_marker)
    gmsh.model.setPhysicalName(1, wall_marker, "Walls")
    gmsh.model.addPhysicalGroup(1, obstacle, obstacle_marker)
    gmsh.model.setPhysicalName(1, obstacle_marker, "Obstacle")
    if not only_dirichlet:
        gmsh.model.addPhysicalGroup(1, inflow, inlet_marker)
        gmsh.model.setPhysicalName(1, inlet_marker, "Inlet")
        gmsh.model.addPhysicalGroup(1, outflow, outlet_marker)
        gmsh.model.setPhysicalName(1, outlet_marker, "Outlet")

# %%
# introduces a subdivision of the arc elements of the circle, uncomment if necessary

## parameters
#h_circle = 0.005            # desired mesh size on circle
#narc = 64                   # number of arc segments
#
## 1) define the points on the circle boundary, capturing their tags
#circle_point_tags = []
#for i in range(narc):
#    theta = 2*np.pi*i/narc
#    x = c_x + r*np.cos(theta)
#    y = c_y + r*np.sin(theta)
#    # 4th argument = nominal mesh size at this point
#    tag = gmsh.model.geo.addPoint(x, y, 0, h_circle)
#    circle_point_tags.append(tag)
#
## 2) connect them with circle‐arcs (so you keep the true curved geometry)
#circle_arc_tags = []
#for i in range(narc):
#    start = circle_point_tags[i]
#    end   = circle_point_tags[(i+1) % narc]
#    # center point of the circle as the arc center
#    center_pt = gmsh.model.geo.addPoint(c_x, c_y, 0, h_circle)
#    arc_tag = gmsh.model.geo.addCircleArc(start, center_pt, end)
#    circle_arc_tags.append(arc_tag)
#
#gmsh.model.geo.synchronize()

# %%
for i in range(60):
    index = i+1
    divisor = i+1
    meshmax = 0.1/divisor
    res_min = r / 3
    if mesh_comm.rank == model_rank:
        distance_field = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(distance_field, "EdgesList", obstacle)
        threshold_field = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(threshold_field, "IField", distance_field)
        gmsh.model.mesh.field.setNumber(threshold_field, "LcMin", res_min)
        gmsh.model.mesh.field.setNumber(threshold_field, "LcMax", 0.25 * H)
        gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", r)
        gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", 2 * H)
        min_field = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", [threshold_field])
        gmsh.model.mesh.field.setAsBackgroundMesh(min_field)

        gmsh.option.setNumber("Mesh.MeshSizeMax", meshmax)

        gmsh.model.mesh.generate(1)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature",       1)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)

        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(1)
        gmsh.model.mesh.optimize("Laplace2D")

        msh, _, ft = gmshio.model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=gdim)
        ft.name = "mesh_tags"

        print("NUMBER OF POINTS : ", msh.geometry.x.shape[0], " (Iteration "+str(i)+")")
        if msh.geometry.x.shape[0] > 100000:
            print("STOPPED AT ITERATION "+str(i)+" BECAUSE NUMBER OF NODES EXCEEDED 100.000")
            break

        filename =  "flow_around_cylinder_mesh_xoff_"+str(x_buffer)
        if only_dirichlet:
            filename = filename + "_only_dirichlet"
        filename = filename +"_"+str(index)+".xdmf"
        with XDMFFile(MPI.COMM_WORLD,filename, "w") as file:
            file.write_mesh(msh)
            file.write_meshtags(ft, msh.geometry)
            file.close()

# %% [markdown]
# ### Plot

# %%
import matplotlib.pyplot as plt
from dolfinx.mesh import compute_incident_entities

plt.plot(msh.geometry.x[:,0], msh.geometry.x[:,1], '+')

n_edge = msh.topology.index_map(1).size_local

colors = ['k.', 'k-', 'r:',  'b-', 'c-', 'y-']

for i in range( n_edge):
    boundary_facets = compute_incident_entities(msh.topology, np.array([i], dtype=np.int32), 1, 0)
    x = msh.geometry.x[boundary_facets, 0]
    y = msh.geometry.x[boundary_facets, 1]
    plt.plot(x, y,  'r-')

for i in range(0, ft.values.shape[0]): 
    boundary_facets = compute_incident_entities(msh.topology, np.array([ft.indices[i]], dtype=np.int32), 1, 0)
    x = msh.geometry.x[boundary_facets, 0]
    y = msh.geometry.x[boundary_facets, 1]

    plt.plot(x, y,  colors[ft.values[i]])

plt.gca().set_aspect(1.0)

# %% [markdown]
# ### Store mesh in file

# %%
filename =  "flow_around_cylinder_mesh_xoff_"+str(x_buffer)
if only_dirichlet:
    filename = filename + "_only_dirichlet"
filename = filename +".xdmf"
with XDMFFile(MPI.COMM_WORLD,filename, "w") as file:
    file.write_mesh(msh)
    file.write_meshtags(ft, msh.geometry)
    file.close()
