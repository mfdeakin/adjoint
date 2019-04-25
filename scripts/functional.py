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
       return None, None
   J_int_approx = u.dot(g) * area
   J_bndry = ((sum(dual_sol.top_bndry_val(), axis=1)
               .dot(primal_sol.top_bndry_deriv())
               + sum(dual_sol.bottom_bndry_val(), axis=1)
               .dot(primal_sol.bottom_bndry_deriv())) * dx
              + (sum(dual_sol.left_bndry_val(), axis=1)
                 .dot(primal_sol.left_bndry_deriv())
                 + sum(dual_sol.right_bndry_val(), axis=1)
                 .dot(primal_sol.right_bndry_deriv())) * dy)
   primal_sol.apply_bc_2()
   du = primal_sol.delta_grid_4()
   du = du.flatten()
   J_delta = du.dot(v) * area
   H_int_approx = v.dot(f) * area
   H_bndry = ((sum(primal_sol.top_bndry_val(), axis=1)
               .dot(dual_sol.top_bndry_deriv())
               + sum(primal_sol.bottom_bndry_val(), axis=1)
               .dot(dual_sol.bottom_bndry_deriv())) * dx
              + (sum(primal_sol.left_bndry_val(), axis=1)
                 .dot(dual_sol.left_bndry_deriv())
                 + sum(primal_sol.right_bndry_val(), axis=1)
                 .dot(dual_sol.right_bndry_deriv())) * dy)
   print(("Approximate Interior J: {:.9f}\n"
          + "Approximate Boundary J: {:.9f}\n"
          + "Delta J: {:.9f}\n"
          + "Approximate Interior H: {:.9f}\n"
          + "Approximate Boundary H: {:.9f}")
         .format(J_int_approx, J_bndry, J_delta, H_int_approx, H_bndry))
   return (J_int_approx + J_bndry,
           J_int_approx + J_bndry - J_delta,
           H_int_approx + H_bndry)
