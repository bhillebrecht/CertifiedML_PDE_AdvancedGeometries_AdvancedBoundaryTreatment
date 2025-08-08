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
# # Compute characteristic parameters of stokes flow around cylinder
#
# ###################################################################################################
#
#  Copyright (c) 2025 Birgit Hillebrecht
#
#  To cite this code in publications, please use
#  
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
import os
try:
    current_dir = os.path.dirname(__file__)
except:
    current_dir = os.getcwd()

import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import pandas as pd 

from scipy.sparse.linalg import splu, spsolve, svds
from scipy.sparse import csr_matrix, csc_matrix, diags

import gmsh
from mpi4py import MPI
from petsc4py import PETSc

from basix.ufl import element
import basix

import ufl 
from ufl import (FacetNormal, Identity, Measure, TestFunction, TrialFunction,
                 as_vector, div, dot, ds, dx, inner, lhs, grad, nabla_grad, rhs, sym, system)

from dolfinx.fem import (Constant, Function, functionspace,
                         assemble_scalar, dirichletbc, form, locate_dofs_topological, set_bc)
from dolfinx.fem.petsc import (apply_lifting, assemble_matrix, assemble_vector,
                               create_vector, create_matrix, set_bc)
from dolfinx.graph import adjacencylist
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
from dolfinx.io import (VTXWriter, distribute_entity_data, gmshio, XDMFFile)
from dolfinx.mesh import create_mesh, meshtags_from_entities
from dolfinx import fem
from dolfinx.mesh import refine, transfer_meshtag, RefinementOption
from dolfinx import mesh, fem


sys.path.append(os.path.join(current_dir, "..", "util"))
sys.path.append(os.path.join(current_dir, ".."))

from xdmf_helper import load_xdmf

gmsh.initialize()

# %% [markdown]
# ### Step 1: Build Meshes

# %%
## number of meshes
n_refine = 10

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

# initialize everything
mesh_comm = MPI.COMM_WORLD
model_rank = 0
if mesh_comm.rank == model_rank:
    rectangle = gmsh.model.occ.addRectangle(x_0, y_0, 0, L+2*x_buffer, H, tag=1)
    obstacle = gmsh.model.occ.addDisk(c_x, c_y, 0, r, r)

# cut the obstacle and synchronize the geometry
if mesh_comm.rank == model_rank:
    fluid = gmsh.model.occ.cut([(gdim, rectangle)], [(gdim, obstacle)])
    gmsh.model.occ.synchronize()

# mark volume elements
fluid_marker = 1
if mesh_comm.rank == model_rank:
    volumes = gmsh.model.getEntities(dim=gdim)
    assert (len(volumes) == 1)
    gmsh.model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], fluid_marker)
    gmsh.model.setPhysicalName(volumes[0][0], fluid_marker, "Fluid")

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

# iterate over different max cell sizes
for i in range(n_refine):
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
            n_refine = i
            break

        filename =  "flow_around_cylinder_mesh_xoff_"+str(x_buffer)
        if only_dirichlet:
            filename = filename + "_only_dirichlet"
        filename = filename +"_"+str(index)+".xdmf"
        filepath = os.path.join(current_dir, "..", "..", "output_data", "flow_around_cylinder", filename)
        with XDMFFile(MPI.COMM_WORLD,filepath, "w") as file:
            file.write_mesh(msh)
            file.write_meshtags(ft, msh.geometry)
            file.close()


# %% [markdown]
# ## Helper functions

# %%
## Somehow the original meshes and facet tags are not correct, so manual identification is performed
def mark_boundary_nodes(mesh):
    dirichlet_boundary = []
    neumann_boundary = []
    is_boundary_node = np.ones(mesh.geometry.x.shape[0], dtype=bool)
    for i in range(mesh.geometry.x.shape[0]):
        if mesh.geometry.x[i, 0]<1e-6:
            dirichlet_boundary.append(i)
        elif 0.41-mesh.geometry.x[i, 1]<1e-6:
            dirichlet_boundary.append(i)
        elif mesh.geometry.x[i, 1]<1e-6:
            dirichlet_boundary.append(i)
        elif (mesh.geometry.x[i, 1]-c_y)**2+(mesh.geometry.x[i, 0]-c_x)**2-r**2<1e-6:
            dirichlet_boundary.append(i)
        elif 1.2-mesh.geometry.x[i, 0]<1e-6:
            neumann_boundary.append(i)
        else:
            is_boundary_node[i] = False
    neumann_boundary = np.array(neumann_boundary)
    dirichlet_boundary = np.array(dirichlet_boundary)
    return dirichlet_boundary, neumann_boundary, is_boundary_node


# %%
def mark_boundary_facets(mesh, is_boundary_node):
    f2v = mesh.topology.connectivity(1,0).array
    f2v = np.reshape(f2v, (int(f2v.shape[0]/2), 2))

    is_boundary_facet = np.zeros(f2v.shape[0], dtype=bool)

    for findex in range(f2v.shape[0]):
        if is_boundary_node[f2v[findex, 0]] and is_boundary_node[f2v[findex, 1]]:
            is_boundary_facet[findex] = True

    return is_boundary_facet

def get_boundary_length(mesh, is_boundary_facet):
    f2v = mesh.topology.connectivity(1,0).array
    f2v = np.reshape(f2v, (int(f2v.shape[0]/2), 2))

    boundary_len = np.zeros(f2v.shape[0])

    for findex in range(f2v.shape[0]):
        if is_boundary_facet[findex]:
            corner1 = mesh.geometry.x[f2v[findex, 0], 0:2]
            corner2 = mesh.geometry.x[f2v[findex, 1], 0:2]
            boundary_len[findex] = np.sqrt(np.sum((corner1 -corner2)**2))

    return boundary_len


# %%
def get_iotas(mesh, neumann_boundary, dirichlet_boundary):
    iota_dirichlet = np.unique(dirichlet_boundary)
    iota_neumann = neumann_boundary

    # remove boundary nodes which are in both boundaries from the neumann bc
    remove_n = []
    for i in range(neumann_boundary.shape[0]):
        if np.any(iota_dirichlet==iota_neumann[i]):
            remove_n.append(i)

    iota_neumann = np.delete(iota_neumann, remove_n)

    iota_all = np.concatenate([iota_dirichlet, iota_neumann], axis=0)

    return iota_dirichlet, iota_neumann, iota_all


# %% [markdown]
# ### Helper functions for triangles

# %%
def triangle_height(x1, y1, x2, y2, x3, y3):
    side_1 = np.array([x2-x1, y2-y1])
    side_2 = np.array([x3-x2, y3-y2])
    side_3 = np.array([x1-x3, y1-y3])

    len_1 = np.sqrt(np.sum(side_1**2))
    len_2 = np.sqrt(np.sum(side_2**2))
    len_3 = np.sqrt(np.sum(side_3**2))
    
    s = 0.5*(len_1 + len_2 + len_3)
    A = np.sqrt(s *(s-len_1)*(s-len_2)*(s-len_3))

    h = 2*A / len_2

    return h

def triangle_height_explicit_base(top, base):
    return triangle_height(top[0], top[1], base[0, 0], base[0, 1], base[1,0], base[1, 1])


# %%
def point_in_triangle(pt, triangle):
    """
    Check if point pt lies inside triangle defined by vertices v1, v2, v3.

    Parameters:
      pt : tuple of floats (x, y)
      v1, v2, v3 : tuples of floats (x, y) for triangle vertices

    Returns:
      True if pt is inside or on the boundary of the triangle; False otherwise.
    """
    x, y = pt
    x1, y1 = triangle[0,0], triangle[0,1]
    x2, y2 = triangle[1,0], triangle[1,1]
    x3, y3 = triangle[2,0], triangle[2,1]

    # Compute vectors
    denom = (y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3)
    if denom == 0:
        # Degenerate triangle
        return False

    # Barycentric coordinates
    a = ((y2 - y3)*(x - x3) + (x3 - x2)*(y - y3)) / denom
    b = ((y3 - y1)*(x - x3) + (x1 - x3)*(y - y3)) / denom
    c = 1 - a - b

    # Check if point is in triangle
    return (0 <= a <= 1) and (0 <= b <= 1) and (0 <= c <= 1)


# %%
def get_gamma_neumann(mesh, iota_neumann):
    gamma_neumann = -1*np.ones((iota_neumann.shape[0], 3))
    c2v = mesh.topology.connectivity(2,0).array
    c2v = np.reshape(c2v, (int(c2v.shape[0]/3), 3))

    for i in range(iota_neumann.shape[0]):
        node_index = iota_neumann[i]
        gamma_neumann[i, 0] = iota_neumann[i]
        
        for j in range(c2v.shape[0]):
            if np.any(c2v[j, :] == node_index):
                top = np.argwhere(c2v[j, :]==node_index)[0][0]
                base = np.squeeze(np.argwhere(c2v[j, :]!=node_index))

                h = triangle_height_explicit_base(mesh.geometry.x[c2v[j, top], 0:2], mesh.geometry.x[c2v[j, base], 0:2])
                check_point = mesh.geometry.x[c2v[j, top], 0:2] + np.array([-0.4*h, 0])

                if point_in_triangle(check_point, mesh.geometry.x[c2v[j, :], 0:2]):
                    gamma_neumann[i, 1] = c2v[j, base[0]]
                    gamma_neumann[i, 2] = c2v[j, base[1]]

    gamma_neumann = np.int64(gamma_neumann)
    return gamma_neumann


# %%
def get_inverse_facet_area_boundary(mesh, iota_all, is_boundary_facet, boundary_len):
    surface_all =np.zeros(iota_all.shape[0])
    mesh.topology.create_connectivity(0,1)
    v2f = mesh.topology.connectivity(0,1)

    for i in range(iota_all.shape[0]):
        node_index = iota_all[i]
        
        neighbor_facets = v2f.links(node_index)
        for k in range(neighbor_facets.shape[0]):
            if is_boundary_facet[neighbor_facets[k]]:
                surface_all[i] += 0.5*boundary_len[neighbor_facets[k]]

    surface_all_inv = 1./surface_all
    return surface_all_inv 


# %%
def get_area_all_cells_per_node(mesh):
    n_coord = mesh.geometry.x.shape[0]

    c2v = mesh.topology.connectivity(2,0).array
    c2v = np.reshape(c2v, (int(c2v.shape[0]/3), 3))

    volume_all = np.zeros(n_coord)
    for j in range(c2v.shape[0]):
        top = 0
        base = [1, 2]
        base_coords = mesh.geometry.x[c2v[j, base]]
        h = triangle_height_explicit_base(mesh.geometry.x[c2v[j, top], 0:2], mesh.geometry.x[c2v[j, base], 0:2])
        A = 0.5*h*np.sqrt((base_coords[0,0]-base_coords[1,0])**2+ (base_coords[1,0]-base_coords[1,1])**2)
        
        for i in range(3):
            volume_all[c2v[j,i]]= A/3.0

    return volume_all 


# %%
def get_D(mesh, iota_dirichlet, gamma_neumann):
    n_neumann = gamma_neumann.shape[0]
    n_dirichlet = iota_dirichlet.shape[0]
    D = sp.csr_matrix((n_dirichlet + n_neumann, mesh.geometry.x.shape[0]))

    # implement Dirichlet in the first n_dirichlet boundaries 
    for i in range(n_dirichlet):
        D[i, iota_dirichlet[i]] = 1.0

    # implement Neumann in the last n_neumann nodes 
    for i in range(n_neumann):
        x1 = mesh.geometry.x[gamma_neumann[i, 0], 0]
        y1 = mesh.geometry.x[gamma_neumann[i, 0], 1]
        x2 = mesh.geometry.x[gamma_neumann[i, 1], 0]
        y2 = mesh.geometry.x[gamma_neumann[i, 1], 1]
        x3 = mesh.geometry.x[gamma_neumann[i, 2], 0]
        y3 = mesh.geometry.x[gamma_neumann[i, 2], 1]

        denom = (x1-x2)*(y2-y3)-(x2-x3)*(y1-y2)
        D[i+n_dirichlet, gamma_neumann[i, 0]] = (y2-y3)/denom
        D[i+n_dirichlet, gamma_neumann[i, 1]] = (-y1 + 2* y2 - y3)/denom
        D[i+n_dirichlet, gamma_neumann[i, 2]] = (y1-y2)/denom

    return D 


# %%
def get_right_inverse(D):
    DDT = np.matmul(D, D.T)
    DDTinv = np.linalg.inv(DDT)
    D0 = np.matmul(D.T, DDTinv)
    return D0

def get_right_inverse_sparse(D_csr):
    G = (D_csr @ D_csr.T).tocsc()
    
    # factorize G once
    lu = splu(G)
    
    # for each column of D we solve G x = D[:, i]
    # and stack the solutions as columns of X = (G^{-1} D)
    m, n = D_csr.shape
    X = np.zeros((m, m), dtype=float)
    I = np.eye(m)

    # convert D to dense columns on the fly and solve
    for i in range(m):
        X[:, i] = lu.solve(I[:, i])                # G x = b

    # now D0 = D^T @ G^{-1}  -> shape (n×m)
    D0 = D_csr.T.dot(csr_matrix(X))
    
    # return as CSC for efficient column‐slicing
    return csc_matrix(D0)


# %%
def get_scaled_matrix_norm(D0, surface_weight, volume_weight):
    return np.max( np.linalg.svdvals(np.matmul(np.matmul(np.diag(volume_weight), D0), np.diag(surface_weight))) )


# %%

def get_scaled_matrix_norm_sparse(D0, surface_weight, volume_weight, tol=1e-6, maxiter=None):
    """
    Compute the spectral norm (largest singular value) of
      diag(volume_weight) @ D0 @ diag(surface_weight)
    using sparse operations throughout.

    Parameters
    ----------
    D0 : scipy.sparse matrix, shape (n, m)
        The right‐inverse or any other sparse operator.
    surface_weight : array_like, length m
        Weights for columns (multiplied on right).
    volume_weight : array_like, length n
        Weights for rows (multiplied on left).
    tol : float, optional
        Convergence tolerance for svds (default 1e-6).
    maxiter : int or None, optional
        Maximum number of iterations for svds (default None, letting ARPACK choose).

    Returns
    -------
    sigma_max : float
        The largest singular value of the weighted operator.
    """
    # build sparse diagonal weight matrices
    Wv = diags(volume_weight)      # shape (n, n)
    Ws = diags(surface_weight)     # shape (m, m)

    # form the weighted operator M
    M = Wv.dot(D0).dot(Ws)         # still sparse

    # compute the top singular value via ARPACK
    # svds returns u, s, vt; we only need s[0]
    # which='LM' asks for the largest magnitude singular values
    u, s, vt = svds(M, k=1, which='LM', tol=tol, maxiter=maxiter)
    return s[0]


# %%
def petsc2array(v):
    s=v.getValues(range(0, v.getSize()[0]), range(0,  v.getSize()[1]))
    return s

def compute_A(domain):
    # 2. Define mixed (velocity-pressure) function space: P2 ⨉ P1
    el_u = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(domain.geometry.dim,))
    el_p = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
    el_mixed = basix.ufl.mixed_element([el_u, el_p])

    # 3. Define trial and test functions
    W = fem.functionspace(domain, el_mixed)
    u, p = ufl.TrialFunctions(W)
    v, q = ufl.TestFunctions(W)

    # 4. Physical parameters
    mu = fem.Constant(domain, PETSc.ScalarType(0.001))  # viscosity = 0.001
    rho = fem.Constant(domain, PETSc.ScalarType(1.0))  # density = 1

    # 5. Define the bilinear form for the Stokes operator
    a = (
        mu * inner(grad(u), grad(v)) * dx                # viscous term
        - div(v) * p * dx                                # pressure term in momentum eq.
        - q * div(u) * dx                                # incompressibility constraint
    )
    
    # 1) make a dolfinx Form for the LHS
    a = form(a)

    # 3) assemble into it
    A2= assemble_matrix( a)
    A2.assemble()

    # getValuesCSR returns (indptr, indices, data)
    indptr, indices, data = A2.getValuesCSR()
    
    # PETSc defines indptr such that len(indptr) = n_rows+1
    n_rows, n_cols = A2.getSize()
    return csr_matrix((data, indices, indptr), shape=(n_rows, n_cols))


# %%
n_nodes = np.zeros(n_refine)
D0_norm = np.zeros(n_refine)
AD0_norm = np.zeros(n_refine)
A0_lrev = np.zeros(n_refine)

for i in range(n_refine):
    filename =  "flow_around_cylinder_mesh_xoff_"+str(x_buffer)+"_"+str(i+1)+".xdmf"
    filepath = os.path.join(current_dir, "..", "..", "output_data", "flow_around_cylinder", filename)
    mesh, facet_tags = load_xdmf(filepath)
    n_nodes[i] = mesh.geometry.x.shape[0]

    # boundary identification and computation
    dirichlet_boundary, neumann_boundary, is_boundary_node = mark_boundary_nodes(mesh)
    is_boundary_facet = mark_boundary_facets(mesh, is_boundary_node)
    boundary_len = get_boundary_length(mesh, is_boundary_facet)
    iota_dirichlet, iota_neumann, iota_all = get_iotas(mesh, neumann_boundary, dirichlet_boundary)
    gamma_neumann = get_gamma_neumann(mesh, iota_neumann)
    
    # weightings
    surface_all_inv = get_inverse_facet_area_boundary(mesh, iota_all, is_boundary_facet, boundary_len)
    volume_all = get_area_all_cells_per_node(mesh)

    # compute D
    D = get_D(mesh, iota_dirichlet, gamma_neumann)
    D0 = get_right_inverse_sparse(D)
    D0_norm[i] = get_scaled_matrix_norm_sparse(D0, surface_all_inv, volume_all)
    
    # compute A
    n_coords = mesh.geometry.x.shape[0]
    A = compute_A(mesh)[0:2*n_coords, 0:2*n_coords]
    _, helper, _ = svds(A, k=1, which='LM', tol=1e-6, maxiter=10000)
    _, helper2, _ = svds(A- helper[0]*sp.eye(n_coords*2), k=1, which='LM', tol=1e-6, maxiter=10000)
    A0_lrev[i] = helper2[0] + helper[0]
    AD0 = A.dot(sp.vstack([D0, D0], format=D0.format))
    AD0_norm[i] = get_scaled_matrix_norm_sparse(AD0, surface_all_inv, np.concatenate([volume_all, volume_all], axis=0))

    print("Mesh no. ", i, " with ", n_nodes[i], " nodes: ", A0_lrev[i],", ", D0_norm[i],", ", AD0_norm[i],", ", sp.linalg.norm(A-A.T), " (omega_n, norm(Dn0), norm(AnDn0), norm(An-An.T))")


# %%
fig, axs = plt.subplots(ncols=1, nrows=3, layout="constrained")

axs[0].plot(n_nodes, A0_lrev, 'k+:')
axs[0].set_xscale("log")
axs[0].set_xticklabels([])
axs[0].set_yticklabels(["0", "0.100", "0.200"])
axs[0].set_yticks([0.000, 0.100, 0.200])
axs[0].set_ylabel("$\omega_n$")

axs[1].plot(n_nodes, D0_norm, 'k+:')
axs[1].set_xscale("log")
axs[1].set_xticklabels([])
axs[1].set_yticklabels(["0.190", "0.180", "0.170"])
axs[1].set_yticks([0.19, 0.18, 0.17])
axs[1].set_ylabel("$\|\mathfrak{D}_{n,0}\|_{L(X_n, U)}$")

axs[2].plot(n_nodes, AD0_norm, 'k+:')
axs[2].set_xscale("log")
axs[2].set_yticklabels(["0.006", "0.004", "0.002", "0"])
axs[2].set_yticks([0.006, 0.004, 0.002, 0])
axs[2].set_ylabel("$\|\mathfrak{A}_n \mathfrak{D}_{n,0}\|_{L(X_n, U)}$")
axs[2].set_xlabel("Number of vertices in mesh $n$")

plt.show()
# %%
df = pd.DataFrame({"n_nodes": n_nodes, "A0_lrev" : A0_lrev,  "D0_norm" : D0_norm, "AD0_norm" : AD0_norm})
df.to_csv(os.path.join(current_dir, "..", "..", "output_data", "flow_around_cylinder", "characteristic_parameters_stokes.csv"))

# %% [markdown]
# ### Clean up artifacts and meshes

# %%
for index in range(n_refine):
    filename =  "flow_around_cylinder_mesh_xoff_"+str(x_buffer)
    if only_dirichlet:
        filename = filename + "_only_dirichlet"
    filename = filename +"_"+str(index+1)
    filepath = os.path.join(current_dir, "..", "..", "output_data", "flow_around_cylinder", filename)
    try:
        os.remove(filepath + ".xdmf")
    except:
        print("Could not remove file: ", filepath + ".xdmf")
    try:
        os.remove(filepath + ".h5")
    except:
        print("Could not remove file: ", filepath + ".h5")

# %% [markdown]
#
