
from multigrid import BoundaryConditions
from matplotlib.pyplot import show
from part1 import convergence, homogen, f, g

def primal_source(x, y, c):
    return ((1.0 + c[1]) * f(x, y) + c[2] * g(x, y))

if __name__ == "__main__":
    functionals = convergence(4, 40, BoundaryConditions(), BoundaryConditions())
    for J_2, J_4, H_2 in functionals:
        print("{:.9f}, {:.9f}, {:.9f}".format(J_2, J_4, H_2))
    show()
