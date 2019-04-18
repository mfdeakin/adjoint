from multigrid import Mesh, BoundaryConditions
from multigrid import (PoissonFVMG_1, PoissonFVMG_2, PoissonFVMG_3, PoissonFVMG_4, PoissonFVMG_5,
                       PoissonFVMG_6, PoissonFVMG_7, PoissonFVMG_8, PoissonFVMG_9, PoissonFVMG_10)
from numpy import sin, pi, nan, isnan, sum, sqrt, array

from matplotlib.pyplot import figure, contour, show, clabel, title, pcolor, colorbar, semilogy, legend
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import time

def plot_3d(x, y, a, name):
    fig = figure()
    ax = fig.gca(projection="3d")
    ax.plot_surface(x, y, a, cmap=cm.gist_heat)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    title(name)

def plot_mesh(m, name):
    plot_3d(m.grid_x()[1:-1, 1:-1], m.grid_y()[1:-1, 1:-1],
            m.array()[1:-1, 1:-1], name)

bc = BoundaryConditions()

def f(x, y):
    return -pi ** 3 * sin(pi * x) * sin(pi * y) / 4.0

primal_solver = PoissonFVMG_4((0.0, 0.0), (1.0,  1.0), 64, 64, bc, f)

plot_mesh(primal_solver.source(), "primal source")
show()

def g(x, y):
    return pi ** 5 * x * (1.0 - x) * y * (1.0 - y) / 2.0


