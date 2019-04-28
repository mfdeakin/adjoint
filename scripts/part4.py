
from multigrid import BoundaryConditions
from matplotlib.pyplot import show
from numpy import array

from functional import choose_solver, run_solver, compute_functional
from part1 import f, g
from part3 import homogen

def primal_source(x, y, c):
    return ((1.0 + c[0]) * f(x, y) + c[1] * g(x, y))

def primal_dalpha(x, y, c):
    return array([f(x, y), g(x, y)])

def check_adjoint_method(cells):
    c = array([0.0, 0.5])
    primal_bc = BoundaryConditions(BoundaryConditions.BC_Type.dirichlet, homogen,
                                   BoundaryConditions.BC_Type.dirichlet, homogen,
                                   BoundaryConditions.BC_Type.dirichlet, homogen,
                                   BoundaryConditions.BC_Type.dirichlet, homogen)
    Solver, levels = choose_solver(cells)
    # This solves for
    # \partial u / \partial c = (\partial L / \partial u)^{-1} \partial J / \partial u
    derivative_solver = Solver((0.0, 0.0), (1.0,  1.0), cells, cells, primal_bc, g)
    area = derivative_solver.dx() * derivative_solver.dy()
    run_solver(derivative_solver, levels)
    psi = derivative_solver.array()[2:-2, 2:-2].flatten() * area

    grid_x = derivative_solver.grid_x()[2:-2, 2:-2].flatten()
    grid_y = derivative_solver.grid_y()[2:-2, 2:-2].flatten()
    dfdc = primal_dalpha(grid_x, grid_y, c)
    dJdc = dfdc.dot(psi)

    eps = 1e-10
    primal_solvers = [Solver((0.0, 0.0), (1.0, 1.0), cells, cells, primal_bc,
                             lambda x, y: primal_source(x, y, c)),
                      Solver((0.0, 0.0), (1.0, 1.0), cells, cells, primal_bc,
                             lambda x, y: primal_source(x, y, c + array([eps, 0.0]))),
                      Solver((0.0, 0.0), (1.0, 1.0), cells, cells, primal_bc,
                             lambda x, y: primal_source(x, y, c + array([0.0, eps])))]
    dual_solver = Solver((0.0, 0.0), (1.0,  1.0), cells, cells, primal_bc, g)
    run_solver(primal_solvers[0], levels)
    run_solver(primal_solvers[1], levels)
    run_solver(primal_solvers[2], levels)
    run_solver(dual_solver, levels)
    _, J0, _ = compute_functional(primal_solvers[0], dual_solver)
    _, J1, _ = compute_functional(primal_solvers[1], dual_solver)
    _, J2, _ = compute_functional(primal_solvers[2], dual_solver)
    print(J0, J1, J2)
    print("Finite Difference value of dJ/dc: {}".format([(J1 - J0) / eps,
                                                         (J2 - J0) / eps]))
    print("Adjoint value of dJ/dc: {}".format(dJdc))

if __name__ == "__main__":
    check_adjoint_method(80)
    check_adjoint_method(160)
    check_adjoint_method(320)
    check_adjoint_method(640)
