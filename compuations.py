from sympy import *
from sympy.calculus import finite_diff
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import symfem
init_session()

os.chdir("/home/dihydromonoxide/Projects/PythonProjects/cymbal_synth/")

from cymbal_msh import *

x, y, z, t = symbols('x y z t')
k = symbols('k', positive=True)
dt = Symbol("\\Delta t")
dt

u = Function('u')(x, y, z, t)
v = Function('v')(x, y, z)

dims = [x, y, z]

eq = integrate(differentiate_finite(u.diff(t, 1), t, points=[t - 2 * dt, t, t - dt]) * v, *((i, -oo, oo) for i in dims)) - k * integrate(sum(u.diff(i) * v.diff(i) for i in dims), *((i, -oo, oo) for i in dims))
eq

class Mesh:

    def __init__(self):
        self.nodes = []
        self.ids = []
        self.faces = []
        self.element = symfem.create_element('triangle', 'vP', 1)
        self.cells = []

    def addNode(self, x, y, z, id):
        assert id == len(self.nodes) + 1
        self.nodes.append([x, y, z])

    def as_numpy(self):
        self.nodes = np.array(self.nodes)
        self.faces = np.array(self.faces)
        self.displacement = np.zeros_like(self.nodes)

    def build(self):
        self.as_numpy()

        for tri in self.faces:
            vs = tuple(self.nodes[i][:2] for i in tri)
            ref = symfem.create_reference("triangle", vertices=vs)
            basis = self.element.map_to_cell(vs)

            self.cells.append((ref, basis))

    def plot(self):
        self.as_numpy()
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        
        t, z, x, y = self.create_triangulation()

        ax.set_box_aspect([np.ptp(i) for i in [x, y, z]])
        ax.plot_trisurf(t, Z=z)


    def addFace(self, idxs, id):
        self.faces.append(np.array([i for i in reversed(idxs[0:3])]) - 1)
        self.faces.append(np.array([i for i in reversed(idxs[3:])]) - 1)

    def create_triangulation(self):
        tris = np.array(self.faces)

        x, y, z = (self.nodes + self.displacement).T
        return Triangulation(x, y, tris), z, x, y

def create_plot_tris():
    mesh = Mesh()
    create_nodes(mesh)
    create_elements(mesh)
    mesh.build()

    mesh.plot()

create_plot_tris()
