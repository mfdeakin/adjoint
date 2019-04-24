from multigrid import Mesh, BoundaryConditions
from multigrid import (PoissonFVMG_1, PoissonFVMG_2, PoissonFVMG_3, PoissonFVMG_4, PoissonFVMG_5,
                       PoissonFVMG_6, PoissonFVMG_7, PoissonFVMG_8, PoissonFVMG_9, PoissonFVMG_10)
from numpy import sin, pi, nan, isnan, sum, sqrt, array, max, min, count_nonzero

from matplotlib.pyplot import figure, contour, show, clabel, title, pcolor, colorbar, semilogy, legend
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import time

def plot_3d(x, y, a, name):
    fig = figure()
    ax = fig.gca(projection="3d")
    a[0, 0] = nan
    a[0, 1] = nan
    a[1, 0] = nan
    a[1, 1] = nan

    a[-1, 0] = nan
    a[-1, 1] = nan
    a[-2, 0] = nan
    a[-2, 1] = nan

    a[0, -1] = nan
    a[0, -2] = nan
    a[1, -1] = nan
    a[1, -2] = nan

    a[-1, -1] = nan
    a[-1, -2] = nan
    a[-2, -1] = nan
    a[-2, -2] = nan
    vmin = min(a[~isnan(a)])
    vmax = max(a[~isnan(a)])
    s = ax.plot_surface(x, y, a, cmap=cm.gist_heat, vmin=vmin, vmax=vmax)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    colorbar(s)
    title(name)

def plot_mesh(m, name):
    plot_3d(m.grid_x(), m.grid_y(),
            m.array(), name)

def run_solver(solver):
    delta = float('inf')
    i = 0
    start = time.time()
    while delta > 1e-12:
        delta = solver.solve_2([(1, 8, 1.45),
                              (2, 4,  1.45),
                              (3, 2,  1.45),
                              (4, 1,  1.45)])
        i += 1
    end = time.time()
    print("Completed in {} iterations, {} s".format(i, end - start))

def f(x, y):
    return -pi ** 3 * sin(pi * x) * sin(pi * y) / 4.0

def g(x, y):
    return pi ** 5 * x * (1.0 - x) * y * (1.0 - y) / 2.0

def compute_functional(primal_sol, dual_sol):
   u = primal_sol.array()[2:-2, 2:-2].flatten()
   v = dual_sol.array()[2:-2, 2:-2].flatten()
   f = primal_sol.source().array().flatten()
   g = dual_sol.source().array().flatten()
   if count_nonzero(isnan(u)) > 0 or count_nonzero(isnan(v)) > 0:
       print("NaN locations in the computed solution")
       return None, None
   J_int_approx = u.dot(g) * primal_sol.dx() * primal_sol.dy()
   quad_off = [-sqrt(3.0 / 20.0), 0.0, -sqrt(3.0 / 20.0)]
   weights = [5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0]
   primal_sol.apply_bc_2()
   du = primal_sol.delta_grid_4()
   du = du.flatten()
   J_delta = du.dot(v) * primal_sol.dx() * primal_sol.dy()
   print("Approximate J: {}\nDelta J: {}".format(J_int_approx, J_delta))
   return J_int_approx, J_int_approx - J_delta

def convergence(depth, cells):
    print(cells)
    primal_solver = PoissonFVMG_4((0.0, 0.0), (1.0,  1.0),
                                  cells, cells, f)

    run_solver(primal_solver)

    dual_solver = PoissonFVMG_4((0.0, 0.0), (1.0,  1.0),
                                cells, cells, g)
    run_solver(dual_solver)

    ug_approx, ug_improved = compute_functional(primal_solver, dual_solver)
    functionals = [(ug_approx, ug_improved)]
    if depth > 1:
        functionals += convergence(depth - 1, cells * 2)
    # if depth < 4:
    #     primal_solver.apply_bc_4()
    #     plot_mesh(primal_solver, "Primal solution at {} cells".format(cells))
    #     dual_solver.apply_bc_4()
    #     plot_mesh(dual_solver, "Dual solution at {} cells".format(cells))
    return functionals

functionals = convergence(2, 40)
for J_2, J_4 in functionals:
    print("{}, {}".format(J_2, J_4))

#show()


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
