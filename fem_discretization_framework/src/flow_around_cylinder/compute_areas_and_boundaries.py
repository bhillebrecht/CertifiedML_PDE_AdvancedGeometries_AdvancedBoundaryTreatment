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

import numpy as np

## Somehow the original meshes and facet tags are not correct, so manual identification is performed
def mark_boundary_nodes(mesh):
    dirichlet_boundary = []
    neumann_boundary = []
    boundary_marking = np.zeros(mesh.geometry.x.shape[0], dtype=int)

    c_x = 0.2
    c_y = 0.2
    r = 0.05
    tol=1e-6

    inlet = 1
    outlet = 2
    wall = 3

    is_boundary_node = np.ones(mesh.geometry.x.shape[0], dtype=bool)
    for i in range(mesh.geometry.x.shape[0]):
        if mesh.geometry.x[i, 0]<tol:
            dirichlet_boundary.append(i)
            boundary_marking[i] = inlet 
        elif 0.41-mesh.geometry.x[i, 1]<tol:
            dirichlet_boundary.append(i)
            boundary_marking[i] = wall
        elif mesh.geometry.x[i, 1]<tol:
            dirichlet_boundary.append(i)
            boundary_marking[i] = wall
        elif (mesh.geometry.x[i, 1]-c_y)**2+(mesh.geometry.x[i, 0]-c_x)**2-r**2<tol:
            dirichlet_boundary.append(i)
            boundary_marking[i] = wall
        elif 1.2-mesh.geometry.x[i, 0]<tol:
            neumann_boundary.append(i)
            boundary_marking[i] = outlet
        else:
            is_boundary_node[i] = False
    neumann_boundary = np.array(neumann_boundary)
    dirichlet_boundary = np.array(dirichlet_boundary)
    return dirichlet_boundary, neumann_boundary, is_boundary_node, boundary_marking

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

def get_facet_area_boundary(mesh, iota_all, is_boundary_facet, boundary_len):
    surface_all =np.zeros(iota_all.shape[0])
    mesh.topology.create_connectivity(0,1)
    mesh.topology.create_connectivity(1,0)
    v2f = mesh.topology.connectivity(0,1)

    for i in range(iota_all.shape[0]):
        node_index = iota_all[i]
        neighbor_facets = v2f.links(node_index)
        for k in range(neighbor_facets.shape[0]):
            if is_boundary_facet[neighbor_facets[k]]:
                surface_all[i] += 0.5*boundary_len[neighbor_facets[k]]

    return surface_all 

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

def get_area_all_cells_per_node_quadrilateral(mesh):
    n_coord = mesh.geometry.x.shape[0]
    c2v = mesh.topology.connectivity(2,0).array
    c2v = np.reshape(c2v, (int(c2v.shape[0]/4), 4))

    volume_all = np.zeros(n_coord)
    for j in range(c2v.shape[0]):
        base = [1, 2]
        base_coords = mesh.geometry.x[c2v[j, base]]
        b = np.sqrt((base_coords[0,0]-base_coords[1,0])**2+ (base_coords[1,0]-base_coords[1,1])**2)

        top = 0
        h = triangle_height_explicit_base(mesh.geometry.x[c2v[j, top], 0:2], mesh.geometry.x[c2v[j, base], 0:2])
        A = 0.5*h*b
        for i in range(3):
            volume_all[c2v[j,i]]= A/3.0

        top = 3
        h = triangle_height_explicit_base(mesh.geometry.x[c2v[j, top], 0:2], mesh.geometry.x[c2v[j, base], 0:2])
        A = 0.5*h*b
        
        for i in range(1, 4):
            volume_all[c2v[j,i]]= A/3.0

    return volume_all 

def get_area_all_cells_per_node_triangular(mesh):
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

def get_area_all_cells_per_node(mesh):
    n = mesh.topology.connectivity(2,0).links(0).shape[0]
    if n == 4:
        return get_area_all_cells_per_node_quadrilateral(mesh)
    else:
        return get_area_all_cells_per_node_triangular(mesh)