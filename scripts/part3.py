
from multigrid import BoundaryConditions
from numpy import sin, pi
from matplotlib.pyplot import show
from functional import convergence
from part1 import f, g

def homogen(x, y):
    return 0.0

if __name__ == "__main__":
    primal_bc = BoundaryConditions(BoundaryConditions.BC_Type.dirichlet, homogen,
                                   BoundaryConditions.BC_Type.dirichlet, homogen,
                                   BoundaryConditions.BC_Type.dirichlet,
                                   lambda x, y: sin(pi * x),
                                   BoundaryConditions.BC_Type.dirichlet, homogen)
    dual_bc = BoundaryConditions(BoundaryConditions.BC_Type.dirichlet, homogen,
                                 BoundaryConditions.BC_Type.dirichlet, homogen,
                                 BoundaryConditions.BC_Type.dirichlet, homogen,
                                 BoundaryConditions.BC_Type.dirichlet,
                                 lambda x, y: x * (1.0 - x))
    functionals = convergence(8, 20, primal_bc, dual_bc, f, g)
    for J_2, J_4, H_2 in functionals:
        print("{:.14f}, {:.14f}, {:.14f}".format(J_2, J_4, H_2))
    show()
