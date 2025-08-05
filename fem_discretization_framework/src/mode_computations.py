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
import logging
from enum import Enum
from scipy.sparse.linalg import eigsh
from scipy.linalg import norm
from scipy.sparse import csr_matrix

import dolfinx

if dolfinx.__version__=="0.9.0":
    from dolfinx import fem, io, mesh, plot
    from dolfinx.fem import Constant, dirichletbc, apply_lifting, set_bc, Function, form
    from dolfinx.fem.petsc import create_vector, assemble_matrix, assemble_vector, LinearProblem
    from dolfinx.mesh import exterior_facet_indices, locate_entities, compute_midpoints, locate_entities_boundary
    from dolfinx.geometry import compute_collisions_points, bb_tree, compute_colliding_cells, compute_closest_entity, create_midpoint_tree
    from dolfinx import io
    from ufl import TrialFunction, TestFunction, inner, dot, grad, dx
    import ufl
    from mpi4py import MPI
    from slepc4py import SLEPc
    from petsc4py import PETSc

    def monitor_EPS_short(
        EPS: SLEPc.EPS, it: int, nconv: int, eig: list, err: list, it_skip: int
    ):
        """
        Concise monitor for EPS.solve().
        Parameters
        ----------
        eps
            Eigenvalue Problem Solver class.
        it
        Current iteration number.
        nconv
        Number of converged eigenvalue.
        eig
        Eigenvalues
        err
        Computed errors.
        it_skip
            Iteration skip.
        """
        if it == 1:
            print("******************************")
            print("***  SLEPc Iterations...   ***")
            print("******************************")
            print("Iter. | Conv. | Max. error")
            print(f"{it:5d} | {nconv:5d} | {max(err):1.1e}")
        elif not it % it_skip:
            print(f"{it:5d} | {nconv:5d} | {max(err):1.1e}")


    def petsc2vector(v):
        s=v.getValues(range(0, v.getSize()))
        return s
    
    def compute_eigenmodes(mesh, n_modes = 30, is_pure_DC= True, boundaries= None, order=2, trace_frequency = 100, mode = 0, xlim=None):
        logging.info("Initialize function space")
        V = fem.functionspace(mesh, ("Lagrange", order))

        # extract relevant dimensions
        tdim = mesh.topology.dim # mesh element dimension
        fdim = tdim - 1 # facet dimension

        logging.info("Initialize boundary conditions")
        if is_pure_DC:
            boundary_facets = exterior_facet_indices(mesh.topology)
            bcs = fem.dirichletbc(0.0, fem.locate_dofs_topological(V, fdim, boundary_facets), V)

        # Compute forms 
        logging.info("Compute Laplace form")
        u = TrialFunction(V)
        v = TestFunction(V)
        a = inner(grad(u), grad(v)) * dx # Laplace operator
        b = inner(u, v) * dx # rhs of the EV problem

        # Assemble matrix and vector
        logging.info("Assemble matrix")
        if is_pure_DC:
            A = assemble_matrix(form(a), bcs = [bcs])
            A.assemble()
            B = assemble_matrix(form(b), bcs = [bcs])
            B.assemble()
        else:
            A = assemble_matrix(form(a), bcs = [])
            A.assemble()
            B = assemble_matrix(form(b), bcs = [])
            B.assemble()

        logging.info("Run SLEPc eigenvalue solver, degrees of freedom "+str(A.getLocalSize()[0]))  
        # get number of indices
        n_An = A.getLocalSize()[0]

        # get number of coordinates 
        points = mesh.geometry.x
        if xlim != None:
            points = points[np.squeeze(np.argwhere(points[:,0]>=xlim[0])), :]
            points = points[np.squeeze(np.argwhere(points[:,0]<=xlim[1])), :]
        n_coord = points.shape[0] 
        
        # Use SLEPc.EPS
        eigensolver = SLEPc.EPS().create(mesh.comm)
        eigensolver.setProblemType(SLEPc.EPS.ProblemType.GNHEP)  # or the right problem type

        eigensolver.setOperators(A, B)
        eigensolver.setDimensions(nev=n_modes)
        if mode == 0:
            eigensolver.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_MAGNITUDE)
        elif mode ==1:
            eigensolver.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_REAL)
        elif mode == 2:
            eigensolver.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
            eigensolver.setTarget(0.0)

            # 4) grab the spectral transform (ST) and turn on shift-invert at sigma=0
            st = eigensolver.getST()
            st.setType(SLEPc.ST.Type.SINVERT)
            st.setShift(0.0)

            # 5) choose a robust direct solver for the inner solves
            ksp = st.getKSP()
            pc  = ksp.getPC()
            pc.setType("lu")            # or "mumps", "superlu"…

        else:
            eigensolver.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_MAGNITUDE)


        eigensolver.setMonitor(
            lambda eps, it, nconv, eig, err: monitor_EPS_short(
                eps, it, nconv, eig, err, trace_frequency
            )
        )
        eigensolver.solve()

        # initialize return value
        if eigensolver.getConverged() < n_modes:
            n_modes = eigensolver.getConverged()
        eigenvalues = np.zeros(n_modes)
        eigenmodes = np.zeros((n_coord, n_modes))
        eigenmodes_in_functionspace = np.zeros((n_An, n_modes))

        # initialize mapping to node values

        tree = bb_tree(mesh, mesh.topology.dim)
        candidates = compute_collisions_points(tree, points)
        cells = compute_colliding_cells(mesh, candidates, points)
        cell_indices = np.empty(points.shape[0], dtype=np.int32)
        for i in range(points.shape[0]):
            coll = cells.links(i)           # array of cell‐indices colliding with point i :contentReference[oaicite:1]{index=1}
            cell_indices[i] = coll[0]  

        # fill return values
        #eigenmode = Function(V)
        for i in range(n_modes): 
            vr = Function(V)
            eigenvalues[i] = (eigensolver.getEigenvalue(i)).real
            eigensolver.getEigenvector(i, vr.x.petsc_vec)
            eigenmodes_in_functionspace[:, i] = vr.x.array[:]
            eigenmodes[:, i:i+1] = vr.eval(points, cell_indices)  

        return eigenvalues, eigenmodes, eigenmodes_in_functionspace
    
    def evaluate_function(points, fun, mesh,  order=2):
        V = fem.functionspace(mesh, ("Lagrange", order))
        n_modes = fun.shape[1]
        n_coord = points.shape[0]

        # find cells
        tree = bb_tree(mesh, mesh.topology.dim)
        candidates = compute_collisions_points(tree, points)
        cells = compute_colliding_cells(mesh, candidates, points)
        cell_indices = []
        kept_indices = []
        for i in range(points.shape[0]):
            coll = cells.links(i)           # array of cell‐indices colliding with point i :contentReference[oaicite:1]{index=1}
            if coll.shape[0] > 0:
                cell_indices.append(coll[0])
                kept_indices.append(i)

        cell_indices = np.array(cell_indices)
        print("Removed " + str(points.shape[0]-cell_indices.shape[0]) +" of " +str(points.shape[0])+ " points.")
        points = points[kept_indices, :]
        n_coord = points.shape[0]

        # initialize return value
        return_val = np.zeros((n_coord, n_modes))

        # fill array with values
        for i in range(n_modes): 
            u = fem.Function(V)
            u.x.array[:] = fun[:, i]
            u.x.scatter_forward()

            return_val[:, i:i+1] = u.eval(points, cell_indices)  

        return return_val, points


    def compute_derivative_information(eigenmodes_in_functionspace, mesh, n_max=100, order=2, direction=0, xlim = None):
        # function sapce setup
        V = fem.functionspace(mesh, ("Lagrange", order))
        V_grad = fem.functionspace(mesh, ("Lagrange", order))

        n_params = eigenmodes_in_functionspace.shape[0]
        n_modes = eigenmodes_in_functionspace.shape[1]

        # initialize mapping to node values
        points = mesh.geometry.x
        if xlim != None:
            points = points[np.squeeze(np.argwhere(points[:,0]>=xlim[0])), :]
            points = points[np.squeeze(np.argwhere(points[:,0]<=xlim[1])), :]
        n_nodes = points.shape[0]

        tree = bb_tree(mesh, mesh.topology.dim)
        candidates = compute_collisions_points(tree, points)
        cells = compute_colliding_cells(mesh, candidates, points)
        cell_indices = np.empty(points.shape[0], dtype=np.int32)
        for i in range(points.shape[0]):
            coll = cells.links(i)           # array of cell‐indices colliding with point i :contentReference[oaicite:1]{index=1}
            cell_indices[i] = coll[0]  

        # initialize returnvalue
        derivative_eigenmodes_in_functionspace = np.zeros((n_params, n_modes))
        derivative_eigenmodes = np.zeros((n_nodes, n_modes))

        # initialize test functions
        v = ufl.TestFunction(V_grad)
        w = ufl.TrialFunction(V_grad)

        # reconstruct function
        for i_mode in range(n_modes):
            u = fem.Function(V)
            u.x.array[:] = eigenmodes_in_functionspace[:,i_mode]
            u.x.scatter_forward()

            a = ufl.inner(w, v) * ufl.dx
            L = ufl.inner(w, ufl.grad(u)[direction] ) * ufl.dx

            problem = fem.petsc.LinearProblem(
                a, L,
                bcs=[]
            )
            dux = problem.solve()      # a dolfinx.fem.Function in V_dx

            # -- 5) extract nodal values as a NumPy array --
            dux_vals = dux.x.array   # length = number of local P1 dofs
            derivative_eigenmodes_in_functionspace[:, i_mode] = dux_vals.copy() 
            derivative_eigenmodes[:, i_mode:i_mode+1] = dux.eval(points, cell_indices)  

        return derivative_eigenmodes, derivative_eigenmodes_in_functionspace

    def get_boundary_and_domain_vertices(mesh, xlim=None):
        """
        Return (boundary_vertices, inside_vertices) as numpy arrays of
        local vertex indices on this MPI rank.
        """
        # 1) find all boundary facets (dim-1 entities) via a trivial predicate
        facet_indices = locate_entities_boundary(
            mesh, mesh.topology.dim - 1,
            # on boundary everywhere
            lambda x: np.full(x.shape[1], True, dtype=bool)
        )

        # 2) ensure we have facet->vertex connectivity
        mesh.topology.create_connectivity(mesh.topology.dim - 1, 0)
        facet_to_vertex = mesh.topology.connectivity(mesh.topology.dim - 1, 0)

        # 2a ) find relevant indices
        if xlim != None:
            relevant_indices = np.intersect1d(np.squeeze(np.argwhere(mesh.geometry.x[:,0]>= xlim[0])), np.squeeze(np.argwhere(mesh.geometry.x[:,0]<=xlim[1])))
            mat_relevance_map = np.zeros(mesh.geometry.x.shape[0])
            for i in range(0, relevant_indices.shape[0]):
                mat_relevance_map[relevant_indices[i]] = i

        # 3) extract all vertices on those facets
        #    `links` maps each input facet index to a slice in `entries`
        boundary_vertices = []
        for f in facet_indices:
            # `.links(f)` returns a list of all vertices attached to facet f
            vs = facet_to_vertex.links(f)
            boundary_vertices.extend(vs)

        boundary_vertices = np.unique(boundary_vertices).astype(np.int32)
        if xlim != None:
            boundary_vertices = mat_relevance_map[boundary_vertices]

        # 4) total set of local vertices
        n_indices = mesh.geometry.x.shape[0]
        if xlim!= None:
            n_indices=relevant_indices.shape[0]
        all_vertices = range(0, n_indices)

        # 5) inside vertices = those not on the boundary
        inside_vertices = np.setdiff1d(all_vertices, boundary_vertices)

        return boundary_vertices, inside_vertices