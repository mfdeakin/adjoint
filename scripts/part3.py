
from multigrid import BoundaryConditions
from numpy import sin, pi
from matplotlib.pyplot import show
from part1 import convergence, homogen

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
    functionals = convergence(4, 40, primal_bc, dual_bc)
    for J_2, J_4, H_2 in functionals:
        print("{:.9f}, {:.9f}, {:.9f}".format(J_2, J_4, H_2))
    show()
