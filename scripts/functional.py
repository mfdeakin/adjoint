from multigrid import (PoissonFVMG_1, PoissonFVMG_2, PoissonFVMG_3, PoissonFVMG_4, PoissonFVMG_5,
                       PoissonFVMG_6, PoissonFVMG_7, PoissonFVMG_8, PoissonFVMG_9, PoissonFVMG_10)

from numpy import nan, isnan, sum, array, max, min, count_nonzero

from matplotlib.pyplot import figure, contour, clabel, title, pcolor, colorbar, semilogy, legend
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

def mg_cycle(levels):
    return [(i + 1, 4 * (levels - i), 1.5) for i in range(levels)]

def run_solver(solver, levels):
    delta = float('inf')
    i = 0
    cycle = mg_cycle(levels)
    start = time.time()
    while delta > 1e-12:
        delta = solver.solve_2(cycle)
        i += 1
    end = time.time()
    print("Completed in {} iterations, {} s".format(i, end - start))

def compute_functional(primal_sol, dual_sol):
   dx = primal_sol.dx()
   dy = primal_sol.dy()
   area = dx * dy
   u = primal_sol.array()[2:-2, 2:-2].flatten()
   v = dual_sol.array()[2:-2, 2:-2].flatten()
   f = primal_sol.source().array().flatten()
   g = dual_sol.source().array().flatten()
   if count_nonzero(isnan(u)) > 0 or count_nonzero(isnan(v)) > 0:
       print("NaN locations in the computed solution")
       return None, None, None
   J_int_approx = u.dot(g) * area
   J_bndry = (dual_sol.top_bndry_val()
              .dot(primal_sol.top_bndry_deriv())
              + dual_sol.bottom_bndry_val()
              .dot(primal_sol.bottom_bndry_deriv())
              + dual_sol.left_bndry_val()
              .dot(primal_sol.left_bndry_deriv())
              + dual_sol.right_bndry_val()
              .dot(primal_sol.right_bndry_deriv()))
   primal_sol.apply_bc_2()
   du = primal_sol.delta_grid_4().flatten()
   J_delta = du.dot(v) * area
   H_int_approx = v.dot(f) * area
   H_bndry = (primal_sol.top_bndry_val()
              .dot(dual_sol.top_bndry_deriv())
              + primal_sol.bottom_bndry_val()
              .dot(dual_sol.bottom_bndry_deriv())
              + primal_sol.left_bndry_val()
              .dot(dual_sol.left_bndry_deriv())
              + primal_sol.right_bndry_val()
              .dot(dual_sol.right_bndry_deriv()))
   print(("Approximate Interior J: {:.9f}\n"
          + "Approximate Boundary J: {:.9f}\n"
          + "Delta J: {:.9f}\n"
          + "Approximate Interior H: {:.9f}\n"
          + "Approximate Boundary H: {:.9f}")
         .format(J_int_approx, J_bndry, J_delta, H_int_approx, H_bndry))
   return (J_int_approx + J_bndry,
           J_int_approx + J_bndry - J_delta,
           H_int_approx + H_bndry)

def choose_solver_helper(cells):
    if cells % 2 == 0:
        return choose_solver_helper(cells // 2) + 1
    else:
        return 0

def choose_solver(cells):
    levels = choose_solver_helper(cells)
    if levels >= 10:
        return PoissonFVMG_10, 10
    else:
        solvers = [PoissonFVMG_1, PoissonFVMG_2, PoissonFVMG_3,
                   PoissonFVMG_4, PoissonFVMG_5, PoissonFVMG_6,
                   PoissonFVMG_7, PoissonFVMG_8, PoissonFVMG_9]
        return solvers[levels - 1], levels

def convergence(depth, cells, primal_bc, dual_bc, primal_source, dual_source):
    print(cells)
    Solver, levels = choose_solver(cells)
    primal_solver = Solver((0.0, 0.0), (1.0,  1.0), cells, cells,
                           primal_bc, primal_source)

    run_solver(primal_solver, levels)

    dual_solver = Solver((0.0, 0.0), (1.0,  1.0), cells, cells,
                         dual_bc, dual_source)
    run_solver(dual_solver, levels)

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
