from typing import Iterable
from sympy import *
from sympy.calculus import finite_diff
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from tqdm import tqdm
from cymbal_msh import *
init_session()

# %%
x, y, z, t = symbols('x y z t')
k = symbols('k', positive=True)
dt = Symbol("\\Delta t")
dt

# %%
u = Function('u')(x, y, z, t)
v = Function('v')(x, y, z)

dims = [x, y, z]

eq = integrate(differentiate_finite(u.diff(t, 1), t, points=[t - 2 * dt, t, t - dt]) * v, *((i, -oo, oo) for i in dims)) - k * integrate(sum(u.diff(i) * v.diff(i) for i in dims), *((i, -oo, oo) for i in dims))
eq

import scipy as sp

class Mesh:

    def __init__(self):
        self.nodes = []
        self.ids = []
        self.faces = []

    def addNode(self, x, y, z, id):
        assert id == len(self.nodes) + 1
        self.nodes.append([x, y, z])

    def as_numpy(self):
        self.nodes = np.array(self.nodes)
        self.faces = np.array(self.faces)

    def plot(self, fig=None, ax=None, offsets=None):
        self.as_numpy()
        fig = fig if fig is not None else plt.figure()
        ax = ax if ax is not None else fig.add_subplot(projection='3d')
        offsets = offsets if offsets is not None else np.zeros_like(self.nodes)
        
        t, z, x, y = self.create_triangulation(offsets)

        ax.set_box_aspect([np.ptp(i) for i in [x, y, z]])
        surf = ax.plot_trisurf(t, Z=z)

        return fig, ax, surf


    def addFace(self, idxs, id):
        for i in range(0, len(idxs), 3):
            self.faces.append(np.array([i for i in idxs[i:i+3]]) - 1)

    def create_triangulation(self, offsets):
        tris = np.array(self.faces)

        x, y, z = (self.nodes + offsets).T
        return Triangulation(x, y, tris), z, x, y

    @staticmethod
    def _reference_basis_function(x, y):
        return [1 - x - y, x, y]

    @staticmethod
    def _get_mat(nodes):
        ex = (nodes[1] - nodes[0])
        ey = (nodes[2] - nodes[0])

        mat = np.array([np.append(ex, 0.0), np.append(ey, .0), np.append(np.linalg.cross(ex, ey), 0), np.append(nodes[0], 1)]).T

        return mat

    def _to_tri(self, tri):
        return tri if not isinstance(tri, int) else self.faces[tri] 

    def _get_nodes(self, tri):
        return tuple(self.nodes[i] for i in tri)
    
    def num_nodes(self):
        return self.nodes.shape[0]

    def num_elements(self):
        return self.faces.shape[0]

    def get_basis(self, tri: Iterable[int] | int, x, y, z):
        tri = self._to_tri(tri)
        nodes = self._get_nodes(tri)

        mat = Mesh._get_mat(nodes)
        lp = np.linalg.inv(mat).dot(np.array(((x), (y), (z), (1))))

        return Mesh._reference_basis_function(*lp[0:2])
    
    @staticmethod
    def _norm(v1, v2, nodes, x=Symbol('x'), y=Symbol('y'), z=Symbol('z')):
        mat = Mesh._get_mat(nodes)
        p = mat.dot(np.array((x, y, 0, 1)))
         
        norm_eq = v1.dot(v2)
        norm_lam = lambdify((x, y), norm_eq.subs({i: p[j] for j, i in enumerate([x, y, z])}))
        norm_eq = sp.integrate.dblquad(norm_lam, 0, 1, lambda x: 0, lambda x: x)[0]

        return norm_eq

    def compute_mat(self, tri):
        x, y, z = symbols('x y z')
        tri = self._to_tri(tri)
        nodes = self._get_nodes(tri)

        M = np.zeros((3, 3))
        K = np.zeros((3, 3))
        basis_funcs = self.get_basis(tri, x, y, z)
        grad_funcs = [np.array([e.diff(s) for e in basis_funcs]) for s in (x, y, z)]
        basis_funcs = list(map(lambda a: np.array((a)), basis_funcs))

        for i in range(3):
            for j in range(3):
                M[i][j] = self._norm(basis_funcs[i], basis_funcs[j], nodes)
                K[i][j] = self._norm(grad_funcs[i], grad_funcs[j], nodes)

        return M, K
def create_mesh():
    mesh = Mesh()
    
    create_nodes(mesh)
    create_elements(mesh)

    # mesh.addNode(0.0, 0.0, 0.0, 1)
    # mesh.addNode(1.0, 0.0, 0.0, 2)
    # mesh.addNode(0.0, 1.0, 0.0, 3)
    # mesh.addFace([0, 1, 2], 1)

    mesh.as_numpy()
    
    return mesh

def test_norm(mesh):

    n = mesh.num_nodes()
    M = np.zeros((n, n))
    K = np.zeros((n, n))

    for tri in tqdm(mesh.faces):
        _M, _K = mesh.compute_mat(tri)

        for li, gi in enumerate(tri):
            for lj, gj in enumerate(tri):
                M[gi][gj] += _M[li][lj]
                K[gi][gj] += _K[li][lj]

    return M, K

import cProfile


mesh = create_mesh()
M, K = test_norm(mesh)

M
K

##

sM = sp.sparse.csc_matrix(M)
sK = sp.sparse.csc_matrix(K)

plt.matshow(sp.sparse.linalg.inv(sM).todense())

def mat_simplification_calc():
    n = mesh.num_nodes()
    u = Function('u')(t)
    Ms, Ks = symbols('M K')

    usm = MatrixSymbol(f'u({t})', n, Integer(1))
    Msm = MatrixSymbol('M', n, n)
    Ksm = MatrixSymbol('K', n, n)

    eq = Ms * u.diff(t, 2) + Ks * u
    eq = eq.replace(Derivative, lambda a, b: Derivative(a, b).as_finite_difference([t, t-dt, t-2 * dt]))
    eq = eq.replace(Function('u')(Wild('w')), lambda w: MatrixSymbol(f'u({w})', n, 1))
    eq = eq.replace(Ms, Msm).replace(Ks, Ksm)
    eq = expand(eq)

    dt2 = Integer(1)/dt**Integer(2)
    eq2lhs = Ksm * usm + dt2 * Msm * usm
    eq2 = expand(Eq(eq2lhs, -eq + eq2lhs))
    eq2
    eq3 = Eq((Ksm + dt2 * Msm) * usm, eq2.rhs)
    eq3
    eq4 = (Ksm + dt2 * Msm).inv() * eq2.rhs
    print(expand(eq4))

def simulate(
        youngs_modulus=(90+120)/2 * 1e9,
        density=(78.8 + 88.8)/2 * 1e3
        ):
    global steps

    steps = []
    
    dt = 1e-4
    c = youngs_modulus / density
    S = sp.sparse.linalg.inv(c * sK + 1/dt**2 * sM).dot(sM) / dt**2
    prev = np.zeros((n))
    p = np.copy(mesh.nodes) - np.array([300.0, 0.0, 0.0])
    prev = 1 / (np.linalg.norm(p, axis=1) * 100 + 1)
    current = np.zeros_like(prev)
    
    num_steps = 1e2
    num_substeps = 1e2
    for _ in tqdm(range(int(num_steps))):
        for _ in range(int(num_substeps)):
            next = 2.0 * S.dot(current) - S.dot(prev)
            np.copyto(prev, current)
            np.copyto(current, next)
        steps.append(np.copy(current))
    
    return steps

import matplotlib.animation as animation

scale = 1e5

fig = plt.figure()

steps = simulate()
steps = [np.array([np.array((0.0, 0.0, j)) for j in i]) for i in steps]
frames = [mesh.plot(fig=fig, offsets=step * scale) for step in steps]

anim = animation.ArtistAnimation(fig, frames)
anim.save('hit.mp4')
