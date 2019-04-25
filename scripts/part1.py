from multigrid import Mesh, BoundaryConditions
from multigrid import (PoissonFVMG_1, PoissonFVMG_2, PoissonFVMG_3, PoissonFVMG_4, PoissonFVMG_5,
                       PoissonFVMG_6, PoissonFVMG_7, PoissonFVMG_8, PoissonFVMG_9, PoissonFVMG_10)
from numpy import sin, pi, sum, sqrt, array, max, min, count_nonzero
from matplotlib.pyplot import show

from functional import plot_mesh, run_solver, compute_functional

def homogen(x, y):
    return 0.0

def f(x, y):
    return -pi ** 3 * sin(pi * x) * sin(pi * y) / 4.0

def g(x, y):
    return pi ** 5 * x * (1.0 - x) * y * (1.0 - y) / 2.0

def convergence(depth, cells, primal_bc, dual_bc, primal_source = f, dual_source = g):
    print(cells)
    primal_solver = PoissonFVMG_4((0.0, 0.0), (1.0,  1.0),
                                  cells, cells, primal_bc, primal_source)

    run_solver(primal_solver)

    dual_solver = PoissonFVMG_4((0.0, 0.0), (1.0,  1.0),
                                cells, cells, dual_bc, dual_source)
    run_solver(dual_solver)

    ug_approx, ug_improved, hf_approx = compute_functional(primal_solver,
                                                           dual_solver)
    functionals = [(ug_approx, ug_improved, hf_approx)]
    if depth > 1:
        functionals += convergence(depth - 1, cells * 2, primal_bc, dual_bc,
                                   primal_source, dual_source)
    if depth < 2:
        primal_solver.apply_bc_4()
        plot_mesh(primal_solver, "Primal solution at {} cells".format(cells))
        dual_solver.apply_bc_4()
        plot_mesh(dual_solver, "Dual solution at {} cells".format(cells))
    return functionals

if __name__ == "__main__":
    bc = BoundaryConditions()
    functionals = convergence(4, 40, bc, bc)
    for J_2, J_4, H_2 in functionals:
        print("{:.9f}, {:.9f}, {:.9f}".format(J_2, J_4, H_2))
    show()

# J = (u_h, g) + (L(u - u_h), v) \approx (u_h, g) + (f - f_h, v_h) + (f - f_h, v - v_h)
# Only the final term is non-computable, but it's of order O(h^4), while the others are O(1) and O(h^2).
# Thus, it's unimportant
# u - u_h = O(h^p)
# v - v_h = O(h^p)
# Compute the 4th order flux integral:
# f_h = L_{h, 4}(u_h)
# f_h - f = O(h^4), g_h - g = O(h^4)


# 1.0005144782158224, 1.0000175227856103
# 1.0001285376947622, 1.0000019232660908
# 1.0000321291776673, 1.0000002233385428

# 1.0005144782158224, 0.9860890625299408
# 1.0001285376947622, 0.99304729910536
# 1.0000321291776673, 0.9965239616332917
