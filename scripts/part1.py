from multigrid import Mesh, BoundaryConditions
from numpy import sin, pi, sum, sqrt, array, max, min, count_nonzero
from matplotlib.pyplot import show

from functional import plot_mesh, convergence

def f(x, y):
    return -pi ** 3 * sin(pi * x) * sin(pi * y) / 4.0

def g(x, y):
    return pi ** 5 * x * (1.0 - x) * y * (1.0 - y) / 2.0

if __name__ == "__main__":
    bc = BoundaryConditions()
    functionals = convergence(6, 20, bc, bc, f, g)
    for J_2, J_4, H_2 in functionals:
        print("{:.14f}, {:.14f}, {:.14f}".format(J_2, J_4, H_2))
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
