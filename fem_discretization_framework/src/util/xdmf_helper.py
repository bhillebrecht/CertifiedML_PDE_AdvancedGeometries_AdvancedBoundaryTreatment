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

from dolfinx.io import XDMFFile
import dolfinx.mesh
from dolfinx.mesh import compute_incident_entities, meshtags
from mpi4py import MPI

import numpy as np


def get_facet_vertex_mappings(mesh):
    n_facet = mesh.topology.index_map(2).size_local

    boundary_facets = np.zeros((n_facet,3), dtype=int)
    for i in range(boundary_facets.shape[0]):
        boundary_facets[i, :] = compute_incident_entities(mesh.topology, i, 2, 0)

    return boundary_facets

def load_xdmf(filename_xdmf, gdim = 3):
    with XDMFFile(MPI.COMM_WORLD, filename_xdmf, "r") as xdmf:
        mesh = xdmf.read_mesh(name="mesh")

    mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)
    with XDMFFile(MPI.COMM_WORLD, filename_xdmf, "r") as xdmf:
        facet_tags = xdmf.read_meshtags(mesh, name="mesh_tags")

    return mesh, facet_tags

def get_topology(mesh):

    c2v = mesh.topology.connectivity(mesh.topology.dim, 0)
    topology = np.reshape(c2v.array, (int(c2v.array.size/4), 4))

    return topology 

def get_geometry(mesh):
    return mesh.geometry.x

def determine_vertex_number(filename_xdmf):
    # import xdmf 
    with XDMFFile(MPI.COMM_WORLD, filename_xdmf, "r") as xdmf:
        mesh = xdmf.read_mesh(name="mesh")
    mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)

    # return vertex number
    return mesh.topology.index_map(0).size_local