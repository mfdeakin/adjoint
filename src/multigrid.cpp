
#include <algorithm>
#include <cmath>
#include <limits>

#include "multigrid.hpp"

Mesh::Mesh(const std::pair<real, real> &corner_1,
           const std::pair<real, real> &corner_2, const size_t cells_x,
           const size_t cells_y, const size_t num_ghostcells) noexcept
    : min_x_(std::min(corner_1.first, corner_2.first)),
      max_x_(std::max(corner_1.first, corner_2.first)),
      min_y_(std::min(corner_1.second, corner_2.second)),
      max_y_(std::max(corner_1.second, corner_2.second)),
      dx_((max_x_ - min_x_) / cells_x),
      dy_((max_y_ - min_y_) / cells_y),
      ghost_cells_(num_ghostcells),
      cva_(std::array<size_t, 2>{cells_x + 2 * ghost_cells_,
                                 cells_y + 2 * ghost_cells_}) {
  assert(!std::isnan(min_x_) && !std::isnan(max_x_));
  assert(!std::isnan(min_y_) && !std::isnan(max_y_));
  assert(min_x_ != max_x_);
  assert(min_y_ != max_y_);
  for(size_t i = 0; i < cva_.shape()[0]; i++) {
    for(size_t j = 0; j < cva_.shape()[1]; j++) {
      cva_(i, j) = 0.0;
    }
  }
}

Mesh::Mesh(const std::pair<real, real> &corner_1,
           const std::pair<real, real> &corner_2, const size_t cells_x,
           const size_t cells_y, std::function<real(real, real)> f,
           const size_t num_ghostcells) noexcept
    : Mesh(corner_1, corner_2, cells_x, cells_y, num_ghostcells) {
  for(int i = -ghost_cells();
      i < static_cast<int>(this->cells_x()) + ghost_cells(); i++) {
    for(int j = -ghost_cells();
        j < static_cast<int>(this->cells_y()) + ghost_cells(); j++) {
      // Use 5th order Gauss-Legendre quadrature to set the cell to the average
      const real x0 = median_x(i), y0 = median_y(j);
      // const real off_x       = dx() / 2.0 * std::sqrt(3.0 / 5.0);
      // const real off_y       = dy() / 2.0 * std::sqrt(3.0 / 5.0);
      constexpr int quad_pts = 1;
      const std::array<real, quad_pts> weights{{1.0}};
      const std::array<real, quad_pts> x_coords{{x0}};
      const std::array<real, quad_pts> y_coords{{y0}};
      // const std::array<real, quad_pts> weights{{5.0 / 18.0, 8.0 / 18.0, 5.0
      // / 18.0}}; const std::array<real, quad_pts> x_coords{{x0 - off_x, x0, x0
      // + off_x}}; const std::array<real, quad_pts> y_coords{{y0 - off_y, y0,
      // y0 + off_y}};
      cv_average(i, j) = 0.0;
      for(int m = 0; m < quad_pts; m++) {
        for(int n = 0; n < quad_pts; n++) {
          cv_average(i, j) +=
              f(x_coords[m], y_coords[n]) * weights[m] * weights[n];
        }
      }
    }
  }
}

// Use bilinear interpolation to estimate the value at the requested point
// Note that we can use this to implement multigrid for mesh size ratios
// other than factors of 2
real Mesh::interpolate(real x, real y) const noexcept {
  const int right_cell  = x_idx(x + dx_ / 2.0);
  const int left_cell   = right_cell - 1;
  const int top_cell    = y_idx(y + dy_ / 2.0);
  const int bottom_cell = top_cell - 1;
  assert(left_cell >= -1);
  assert(right_cell <= cells_x());
  assert(bottom_cell >= -1);
  assert(top_cell <= cells_y());
  const real left_weight   = (median_x(right_cell) - x) / dx_;
  const real bottom_weight = (median_y(top_cell) - y) / dy_;
  return left_weight *
             (bottom_weight * cv_average(left_cell, bottom_cell) +
              (1.0 - bottom_weight) * cv_average(left_cell, top_cell)) +
         (1.0 - left_weight) *
             (bottom_weight * cv_average(right_cell, bottom_cell) +
              (1.0 - bottom_weight) * cv_average(right_cell, top_cell));
}

BoundaryConditions::BoundaryConditions(
    const BC_Type left_t, const std::function<real(real, real)> &left_bc,
    const BC_Type right_t, const std::function<real(real, real)> &right_bc,
    const BC_Type top_t, const std::function<real(real, real)> &top_bc,
    const BC_Type bottom_t, const std::function<real(real, real)> &bottom_bc)
    : left_t_(left_t),
      right_t_(right_t),
      top_t_(top_t),
      bottom_t_(bottom_t),
      left_bc_(left_bc),
      right_bc_(right_bc),
      top_bc_(top_bc),
      bottom_bc_(bottom_bc) {}

std::array<std::array<real, 2>, 4> BoundaryConditions::construct_basis(
    const real delta) const noexcept {
  // First note that since we're writing a cubic polynomial in terms of x and y,
  // and one of the coordinates is unchanging, we only need to compute the terms
  // which are changing.
  // We want a set of 4 polynomials which have an average value of either 1 or 0
  // in the 3 interior cells closest to the boundary, and which have a value of
  // either 0 or 1 at the boundary.
  //
  // Our system of equations in matrix form is
  // [[0          0          0       1]  [ - a_i - ]
  //  [dx^3/4     dx^2/3     dx/2    1]  [ - b_i - ]
  //  [15/4 dx^3  7/3 dx^2   3/2 dx  1]  [ - c_i - ] = I
  //  [65/4 dx^3  19/3 dx^2  5/2 dx  1]] [ - d_i - ]
  // Our basis is given by inverting this matrix, which is
  // [[-2/(3 dx^3)  11/(9 dx^4)  -7/(9 dx^4)    2/(9 dx^4)]
  //  [3/dx^2      -5 / dx^3      5/(2 dx^3)   -1/(2 dx^3)]
  //  [-11/(3 dx)   85/(18 dx^2) -23/(18 dx^2)  2/(9 dx^2)]
  //  [1            0             0             0]]
  //
  // Each column is one of our polynomials coefficients,
  // starting at the top with the highest exponent,
  // ending at the bottom with the smallest exponent
  Numerical::Polynomial<real, 3, 1> basis_poly;
  // This is polynomial is 1 at the boundary midpoint and has 0 average value in
  // all the control volumes
  basis_poly.coeff({3}) = -2.0 / (delta * delta * delta * 3.0);
  basis_poly.coeff({2}) = 3.0 / (delta * delta);
  basis_poly.coeff({1}) = -11.0 / (delta * 3.0);
  basis_poly.coeff({0}) = 1.0;
  std::array<std::array<real, 2>, 4> basis_coeffs;
  basis_coeffs[0][0] = (basis_poly.integrate(0).eval(0.0) -
                        basis_poly.integrate(0).eval(-delta)) /
                       delta;
  basis_coeffs[0][1] = (basis_poly.integrate(0).eval(-delta) -
                        basis_poly.integrate(0).eval(-2.0 * delta)) /
                       delta;

  basis_poly.coeff({3}) = 11.0 / (delta * delta * delta * 9.0);
  basis_poly.coeff({2}) = -5.0 / (delta * delta);
  basis_poly.coeff({1}) = 85.0 / (delta * 18.0);
  basis_poly.coeff({0}) = 0.0;
  basis_coeffs[1][0]    = (basis_poly.integrate(0).eval(0.0) -
                        basis_poly.integrate(0).eval(-delta)) /
                       delta;
  basis_coeffs[1][1] = (basis_poly.integrate(0).eval(-delta) -
                        basis_poly.integrate(0).eval(-2.0 * delta)) /
                       delta;

  basis_poly.coeff({3}) = -7.0 / (delta * delta * delta * 9.0);
  basis_poly.coeff({2}) = 5.0 / (delta * delta * 2.0);
  basis_poly.coeff({1}) = -23.0 / (delta * 18.0);
  basis_poly.coeff({0}) = 0.0;
  basis_coeffs[2][0]    = (basis_poly.integrate(0).eval(0.0) -
                        basis_poly.integrate(0).eval(-delta)) /
                       delta;
  basis_coeffs[2][1] = (basis_poly.integrate(0).eval(-delta) -
                        basis_poly.integrate(0).eval(-2.0 * delta)) /
                       delta;

  basis_poly.coeff({3}) = 2.0 / (delta * delta * delta * 9.0);
  basis_poly.coeff({2}) = -1.0 / (delta * delta * 2.0);
  basis_poly.coeff({1}) = 2.0 / (delta * 9.0);
  basis_poly.coeff({0}) = 0.0;
  basis_coeffs[3][0]    = (basis_poly.integrate(0).eval(0.0) -
                        basis_poly.integrate(0).eval(-delta)) /
                       delta;
  basis_coeffs[3][1] = (basis_poly.integrate(0).eval(-delta) -
                        basis_poly.integrate(0).eval(-2.0 * delta)) /
                       delta;

  basis_coeffs[0][0] = 4.0;
  basis_coeffs[1][0] = -13.0 / 3.0;
  basis_coeffs[2][0] = 5.0 / 3.0;
  basis_coeffs[3][0] = -1.0 / 3.0;

  basis_coeffs[0][1] = 16.0;
  basis_coeffs[1][1] = -70.0 / 3.0;
  basis_coeffs[2][1] = 32.0 / 3.0;
  basis_coeffs[3][1] = -7.0 / 3.0;

  return basis_coeffs;
}

template <>
void BoundaryConditions::apply<4>(Mesh &mesh) const noexcept {
  // Uses a cubic interpolating polynomial with no cross terms to interpolate
  // the values of the ghost cells, giving a 4th order implementation of the
  // boundary conditions
  //
  // Currently only implemented for Dirichlet boundary conditions
  assert(bottom_t_ == BC_Type::dirichlet);
  assert(top_t_ == BC_Type::dirichlet);
  assert(left_t_ == BC_Type::dirichlet);
  assert(right_t_ == BC_Type::dirichlet);
  const auto basis_coeffs = construct_basis(1.0);
  // Set the top and bottom ghost cells
  for(int i = 0; i < mesh.cells_x(); i++) {
    const real x = mesh.median_x(i);
    {
      const real y_bottom = mesh.bottom_y(0);

      mesh[{i, -2}] = basis_coeffs[0][1] * bottom_bc_(x, y_bottom) +
                      basis_coeffs[1][1] * mesh[{i, 0}] +
                      basis_coeffs[2][1] * mesh[{i, 1}] +
                      basis_coeffs[3][1] * mesh[{i, 2}];
      mesh[{i, -1}] = basis_coeffs[0][0] * bottom_bc_(x, y_bottom) +
                      basis_coeffs[1][0] * mesh[{i, 0}] +
                      basis_coeffs[2][0] * mesh[{i, 1}] +
                      basis_coeffs[3][0] * mesh[{i, 2}];
    }
    {
      const real y_top = mesh.bottom_y(mesh.cells_y());

      mesh[{i, mesh.cells_y()}] =
          basis_coeffs[0][0] * top_bc_(x, y_top) +
          basis_coeffs[1][0] * mesh[{i, mesh.cells_y() - 1}] +
          basis_coeffs[2][0] * mesh[{i, mesh.cells_y() - 2}] +
          basis_coeffs[3][0] * mesh[{i, mesh.cells_y() - 3}];
      mesh[{i, mesh.cells_y() + 1}] =
          basis_coeffs[0][1] * top_bc_(x, y_top) +
          basis_coeffs[1][1] * mesh[{i, mesh.cells_y() - 1}] +
          basis_coeffs[2][1] * mesh[{i, mesh.cells_y() - 2}] +
          basis_coeffs[3][1] * mesh[{i, mesh.cells_y() - 3}];
    }
  }
  for(int j = 0; j < mesh.cells_y(); j++) {
    const real y = mesh.median_y(j);
    {
      const real x_left = mesh.left_x(0);

      mesh[{-2, j}] = basis_coeffs[0][1] * left_bc_(x_left, y) +
                      basis_coeffs[1][1] * mesh[{0, j}] +
                      basis_coeffs[2][1] * mesh[{1, j}] +
                      basis_coeffs[3][1] * mesh[{2, j}];
      mesh[{-1, j}] = basis_coeffs[0][1] * left_bc_(x_left, y) +
                      basis_coeffs[1][1] * mesh[{0, j}] +
                      basis_coeffs[2][1] * mesh[{1, j}] +
                      basis_coeffs[3][1] * mesh[{2, j}];
    }
    {
      const real x_right = mesh.left_x(mesh.cells_x());

      mesh[{mesh.cells_x() + 1, j}] =
          basis_coeffs[0][1] * right_bc_(x_right, y) +
          basis_coeffs[1][1] * mesh[{mesh.cells_x() - 1, j}] +
          basis_coeffs[2][1] * mesh[{mesh.cells_x() - 2, j}] +
          basis_coeffs[3][1] * mesh[{mesh.cells_x() - 3, j}];
      mesh[{mesh.cells_x(), j}] =
          basis_coeffs[0][0] * right_bc_(x_right, y) +
          basis_coeffs[1][0] * mesh[{mesh.cells_x() - 1, j}] +
          basis_coeffs[2][0] * mesh[{mesh.cells_x() - 2, j}] +
          basis_coeffs[3][0] * mesh[{mesh.cells_x() - 3, j}];
    }
  }
  // Prolongating with bilinear interpolation requires the corner ghost cell to
  // be a reasonable approximation of its actual value - set it to the value of
  // the internal corner cell. Better might be using the inner corner cell
  // gradient to estimate it
  mesh[{-1, -1}]             = mesh[{0, 0}];
  mesh[{-1, mesh.cells_y()}] = mesh[{0, mesh.cells_y() - 1}];
  mesh[{mesh.cells_x(), -1}] = mesh[{mesh.cells_x() - 1, 0}];
  mesh[{mesh.cells_x(), mesh.cells_y()}] =
      mesh[{mesh.cells_x() - 1, mesh.cells_y() - 1}];
}

template <>
void BoundaryConditions::apply<2>(Mesh &mesh) const noexcept {
  // TODO: Reduce code duplication here...
  // Each loop is over a different boundary, they set the ghost cells based on
  // the type of boundary conidtion and the current system
  auto apply_dirichlet = [](real &ghost_cell, const real bndry_cell,
                            const real bc_value) {
    ghost_cell = 2.0 * bc_value - bndry_cell;
  };
  auto apply_neumann = [](real &ghost_cell, const real bndry_cell,
                          const real bc_value, const real h) {
    ghost_cell = h * bc_value + bndry_cell;
  };
  for(int i = 0; i < mesh.cells_x(); i++) {
    const real y_bottom  = mesh.bottom_y(0);
    const real x         = mesh.median_x(i);
    const real bndry_val = mesh[{i, 0}];
    real &ghost_cell     = mesh[{i, -1}];
    if(bottom_t_ == BC_Type::dirichlet) {
      apply_dirichlet(ghost_cell, bndry_val, bottom_bc_(x, y_bottom));
    } else {
      apply_neumann(ghost_cell, bndry_val, bottom_bc_(x, y_bottom), mesh.dy());
    }
    for(int k = 1; k < mesh.ghost_cells(); k++) {
      // Continue the linear extrapolation
      mesh[{i, -1 - k}] =
          (mesh[{i, -1}] - mesh[{i, 0}]) * (1.0 + k) + mesh[{i, 0}];
    }
  }
  for(int i = 0; i < mesh.cells_x(); i++) {
    const real y_top     = mesh.top_y(mesh.cells_y() - 1);
    const real x         = mesh.median_x(i);
    const real bndry_val = mesh[{i, mesh.cells_y() - 1}];
    real &ghost_cell     = mesh[{i, mesh.cells_y()}];
    if(top_t_ == BC_Type::dirichlet) {
      apply_dirichlet(ghost_cell, bndry_val, top_bc_(x, y_top));
    } else {
      apply_neumann(ghost_cell, bndry_val, top_bc_(x, y_top), mesh.dy());
    }
    for(int k = 1; k < mesh.ghost_cells(); k++) {
      // Continue the linear extrapolation
      mesh[{i, mesh.cells_y() + k}] =
          (mesh[{i, mesh.cells_y()}] - mesh[{i, mesh.cells_y() - 1}]) *
              (1.0 + k) +
          mesh[{i, mesh.cells_y() - 1}];
    }
  }

  for(int j = 0; j < mesh.cells_y(); j++) {
    const real x_left    = mesh.left_x(0);
    const real y         = mesh.median_y(j);
    const real bndry_val = mesh[{0, j}];
    real &ghost_cell     = mesh[{-1, j}];
    if(left_t_ == BC_Type::dirichlet) {
      apply_dirichlet(ghost_cell, bndry_val, left_bc_(x_left, y));
    } else {
      apply_neumann(ghost_cell, bndry_val, left_bc_(x_left, y), mesh.dx());
    }
    for(int k = 1; k < mesh.ghost_cells(); k++) {
      // Continue the linear extrapolation
      mesh[{-1 - k, j}] =
          (mesh[{-1, j}] - mesh[{0, j}]) * (1.0 + k) + mesh[{0, j}];
    }
  }
  for(int j = 0; j < mesh.cells_y(); j++) {
    const real x_right   = mesh.right_x(mesh.cells_x() - 1);
    const real y         = mesh.median_y(j);
    const real bndry_val = mesh[{mesh.cells_x() - 1, j}];
    real &ghost_cell     = mesh[{mesh.cells_x(), j}];
    if(right_t_ == BC_Type::dirichlet) {
      apply_dirichlet(ghost_cell, bndry_val, right_bc_(x_right, y));
    } else {
      apply_neumann(ghost_cell, bndry_val, right_bc_(x_right, y), mesh.dx());
    }
    for(int k = 1; k < mesh.ghost_cells(); k++) {
      // Continue the linear extrapolation
      mesh[{mesh.cells_x() + k, j}] =
          (mesh[{mesh.cells_x(), j}] - mesh[{mesh.cells_x() - 1, j}]) *
              (1.0 + k) +
          mesh[{mesh.cells_x() - 1, j}];
    }
  }
}

// Use this to enable initializing the homogeneous system easy
constexpr real zero_source(real, real) noexcept { return 0.0; }

PoissonFVMGSolverBase::PoissonFVMGSolverBase(
    const std::pair<real, real> &corner_1,
    const std::pair<real, real> &corner_2, const size_t cells_x,
    const size_t cells_y) noexcept
    : PoissonFVMGSolverBase(corner_1, corner_2, cells_x, cells_y, zero_source) {
}

PoissonFVMGSolverBase::PoissonFVMGSolverBase(
    const std::pair<real, real> &corner_1,
    const std::pair<real, real> &corner_2, const size_t cells_x,
    const size_t cells_y, const BoundaryConditions &bc) noexcept
    : PoissonFVMGSolverBase(corner_1, corner_2, cells_x, cells_y, bc,
                            zero_source) {}

PoissonFVMGSolverBase::PoissonFVMGSolverBase(
    const std::pair<real, real> &corner_1,
    const std::pair<real, real> &corner_2, const size_t cells_x,
    const size_t cells_y,
    const std::function<real(real, real)> &source) noexcept
    : PoissonFVMGSolverBase(corner_1, corner_2, cells_x, cells_y,
                            BoundaryConditions(), source) {}

PoissonFVMGSolverBase::PoissonFVMGSolverBase(
    const std::pair<real, real> &corner_1,
    const std::pair<real, real> &corner_2, const size_t cells_x,
    const size_t cells_y, const BoundaryConditions &bc,
    const std::function<real(real, real)> &source) noexcept
    : Mesh(corner_1, corner_2, cells_x, cells_y, 2),
      bc_(bc),
      source_(corner_1, corner_2, cells_x, cells_y, source, 0) {}

template <>
real PoissonFVMGSolverBase::delta<2>(const int i, const int j) const noexcept {
  // Computes the difference between the Laplacian and the source term
  assert(i >= 0);
  assert(i < cells_x());
  assert(j >= 0);
  assert(j < cells_y());
  // Our problem is the form of \del u = f
  // The residual is then just r = \del u - f
  return ((cv_average(i - 1, j) - 2.0 * cv_average(i, j) +
           cv_average(i + 1, j)) /
              (dx_ * dx_) +
          (cv_average(i, j - 1) - 2.0 * cv_average(i, j) +
           cv_average(i, j + 1)) /
              (dy_ * dy_)) -
         source_[{i, j}];
}

template <>
real PoissonFVMGSolverBase::delta<4>(const int i, const int j) const noexcept {
  // Computes the difference between the Laplacian and the source term
  assert(i >= 0);
  assert(i < cells_x());
  assert(j >= 0);
  assert(j < cells_y());
  // Our problem is the form of \del u = f
  // The residual is then just r = \del u - f
  real delta_x = s_nan;
  if(i >= 2 && i < cells_x() - 2) {
    delta_x = -cv_average(i - 2, j) + 16.0 * cv_average(i - 1, j) -
              30.0 * cv_average(i, j) + 16.0 * cv_average(i + 1, j) -
              cv_average(i + 2, j);

  } else if(i == 1) {
    delta_x = -cv_average(i + 3, j) + 4.0 * cv_average(i + 2, j) +
              6.0 * cv_average(i + 1, j) - 20.0 * cv_average(i, j) +
              11.0 * cv_average(i - 1, j);
  } else if(i == cells_x() - 2) {
    delta_x = -cv_average(i - 3, j) + 4.0 * cv_average(i - 2, j) +
              6.0 * cv_average(i - 1, j) - 20.0 * cv_average(i, j) +
              11.0 * cv_average(i + 1, j);

  } else if(i == 0) {
    delta_x = 11.0 * cv_average(i + 4, j) - 56.0 * cv_average(i + 3, j) +
              114.0 * cv_average(i + 2, j) - 104.0 * cv_average(i + 1, j) +
              35.0 * cv_average(i, j);
  } else if(i == cells_x() - 1) {
    delta_x = 11.0 * cv_average(i - 4, j) - 56.0 * cv_average(i - 3, j) +
              114.0 * cv_average(i - 2, j) - 104.0 * cv_average(i - 1, j) +
              35.0 * cv_average(i, j);
  }
  delta_x /= 12.0 * dx_ * dx_;
  real delta_y = s_nan;
  if(j >= 2 && j < cells_y() - 2) {
    delta_y = -cv_average(i, j - 2) + 16.0 * cv_average(i, j - 1) -
              30.0 * cv_average(i, j) + 16.0 * cv_average(i, j + 1) -
              cv_average(i, j + 2);

  } else if(j == 1) {
    delta_y = -cv_average(i, j + 3) + 4.0 * cv_average(i, j + 2) +
              6.0 * cv_average(i, j + 1) - 20.0 * cv_average(i, j) +
              11.0 * cv_average(i, j - 1);
  } else if(j == cells_y() - 2) {
    delta_y = -cv_average(i, j - 3) + 4.0 * cv_average(i, j - 2) +
              6.0 * cv_average(i, j - 1) - 20.0 * cv_average(i, j) +
              11.0 * cv_average(i, j + 1);

  } else if(j == 0) {
    delta_y = 11.0 * cv_average(i, j + 4) - 56.0 * cv_average(i, j + 3) +
              114.0 * cv_average(i, j + 2) - 104.0 * cv_average(i, j + 1) +
              35.0 * cv_average(i, j);
  } else if(j == cells_y() - 1) {
    delta_y = 11.0 * cv_average(i, j - 4) - 56.0 * cv_average(i, j - 3) +
              114.0 * cv_average(i, j - 2) - 104.0 * cv_average(i, j - 1) +
              35.0 * cv_average(i, j);
  }
  delta_y /= 12.0 * dy_ * dy_;
  return delta_x + delta_y - source_[{i, j}];
}

int PoissonFVMGSolverBase::cell_index(const int i, const int j) const noexcept {
  assert(i >= 0);
  assert(i < cells_x());
  assert(j >= 0);
  assert(j < cells_y());
  return i * cells_y() + j;
}

template <>
matrix PoissonFVMGSolverBase::operator_mtx<2>() const noexcept {
  const unsigned int N = static_cast<unsigned long>(cells_x()) *
                         static_cast<unsigned long>(cells_y());
  matrix L(matrix::shape_type{{N, N}});
  for(unsigned int i = 0; i < N; i++) {
    for(unsigned int j = 0; j < N; j++) {
      L(i, j) = 0.0;
    }
  }
  const real inv_dxsq = 1.0 / (dx() * dx());
  const real inv_dysq = 1.0 / (dy() * dy());
  for(int i = 0; i < cells_x(); i++) {
    for(int j = 0; j < cells_y(); j++) {
      const int idx = cell_index(i, j);
      if(i > 0) {
        const int left = cell_index(i - 1, j);
        L(idx, left)   = inv_dxsq;
      }
      if(j > 0) {
        const int below = cell_index(i, j - 1);
        L(idx, below)   = inv_dysq;
      }
      L(idx, idx) = -2.0 * (inv_dxsq + inv_dysq);
      if(j < cells_y() - 1) {
        const int above = cell_index(i, j + 1);
        L(idx, above)   = inv_dysq;
      }
      if(i < cells_x() - 1) {
        const int right = cell_index(i + 1, j);
        L(idx, right)   = inv_dxsq;
      }
    }
  }
  return L;
}

template <>
matrix PoissonFVMGSolverBase::operator_mtx<4>() const noexcept {
  const unsigned int N = static_cast<unsigned long>(cells_x()) *
                         static_cast<unsigned long>(cells_y());
  matrix L(matrix::shape_type{{N, N}});
  for(unsigned int i = 0; i < N; i++) {
    for(unsigned int j = 0; j < N; j++) {
      L(i, j) = 0.0;
    }
  }
  const real inv_dxsq = 1.0 / (12.0 * dx() * dx());
  const real inv_dysq = 1.0 / (12.0 * dy() * dy());
  for(int i = 0; i < cells_x(); i++) {
    for(int j = 0; j < cells_y(); j++) {
      const int idx = cell_index(i, j);
      if(i > 0) {
        if(i > 1) {
          const int left_2 = cell_index(i - 2, j);
          L(idx, left_2)   = -inv_dxsq;
        }
        const int left = cell_index(i - 1, j);
        L(idx, left)   = 16.0 * inv_dxsq;
      }
      if(j > 0) {
        if(j > 1) {
          const int below_2 = cell_index(i, j - 2);
          L(idx, below_2)   = -inv_dysq;
        }
        const int below = cell_index(i, j - 1);
        L(idx, below)   = 16.0 * inv_dysq;
      }
      L(idx, idx) = -30.0 * (inv_dxsq + inv_dysq);
      if(j > 0) {
        const int above = cell_index(i, j + 1);
        L(idx, above)   = 16.0 * inv_dysq;
        if(j < cells_y() - 1) {
          const int above_2 = cell_index(i, j + 2);
          L(idx, above_2)   = -inv_dysq;
        }
      }
      if(i < cells_x()) {
        const int right = cell_index(i + 1, j);
        L(idx, right)   = 16.0 * inv_dxsq;
        if(i < cells_x() - 1) {
          const int right_2 = cell_index(i + 2, j);
          L(idx, right_2)   = -inv_dxsq;
        }
      }
    }
  }
  return L;
}

template <unsigned int order>
real PoissonFVMGSolverBase::poisson_pgs_or(const real or_term) noexcept {
  real max_diff = -std::numeric_limits<real>::infinity();
  bc_.apply<order>(*this);
  const real diff_scale =
      (dx_ * dx_ * dy_ * dy_ / (2.0 * (dx_ * dx_ + dy_ * dy_)));
  for(int i = 0; i < cells_x(); i++) {
    for(int j = 0; j < cells_y(); j++) {
      const real diff = or_term * diff_scale * delta<order>(i, j);
      max_diff        = std::max(max_diff, std::abs(diff));
      cv_average(i, j) += diff;
    }
  }
  return max_diff;
}

void PoissonFVMGSolverBase::restrict(
    const PoissonFVMGSolverBase &src) noexcept {
  // This computes a second order restriction of the residual for the coarser
  // mesh
  // We're assuming the cells are uniform
  assert(src.cells_x() == 2 * cells_x());
  assert(src.cells_y() == 2 * cells_y());

  for(int i = 0; i < cells_x(); i++) {
    for(int j = 0; j < cells_y(); j++) {
      // The restriction sets the source to the average of the residual for
      // the cells in the same area
      source_[{i, j}] = -0.25 * (src.template delta<2>(2 * i, 2 * j) +
                                 src.template delta<2>(2 * i, 2 * j + 1) +
                                 src.template delta<2>(2 * i + 1, 2 * j) +
                                 src.template delta<2>(2 * i + 1, 2 * j + 1));
      // It's not clear how to best precondition the coarser grids, so just
      // set it to 0, it appears to work reasonably well
      cv_average(i, j) = 0.0;
    }
  }
}

real PoissonFVMGSolverBase::prolongate(Mesh &dest) const noexcept {
  assert(dest.cells_x() == 2 * cells_x());
  assert(dest.cells_y() == 2 * cells_y());
  real max_diff = -std::numeric_limits<real>::infinity();
  // Because we the values in our boundaries are incomplete, only use
  // interpolation to set the cells which aren't next to the boundaries
  for(int i = 0; i < dest.cells_x(); i++) {
    const real x = dest.median_x(i);
    for(int j = 0; j < dest.cells_y(); j++) {
      const real y    = dest.median_y(j);
      const real diff = interpolate(x, y);
      // const real diff = cv_average(i / 2, j / 2);
      max_diff = std::max(max_diff, std::abs(diff));
      dest[{i, j}] += diff;
    }
  }
  return max_diff;
}

xt::xtensor<real, 1> PoissonFVMGSolverBase::left_bndry_deriv() const noexcept {
  xt::xtensor<real, 1> deriv(xt::xtensor<real, 1>::shape_type{
      {static_cast<unsigned long>(cells_y())}});
  for(int j = 0; j < cells_y(); j++) {
    deriv(j) = (cv_average(-1, j) - cv_average(0, j)) / dx();
  }
  return deriv;
}

xt::xtensor<real, 1> PoissonFVMGSolverBase::right_bndry_deriv() const noexcept {
  xt::xtensor<real, 1> deriv(xt::xtensor<real, 1>::shape_type{
      {static_cast<unsigned long>(cells_y())}});
  for(int j = 0; j < cells_y(); j++) {
    deriv(j) = (cv_average(cells_x(), j) - cv_average(cells_x() - 1, j)) / dx();
  }
  return deriv;
}

xt::xtensor<real, 1> PoissonFVMGSolverBase::bottom_bndry_deriv() const
    noexcept {
  xt::xtensor<real, 1> deriv(xt::xtensor<real, 1>::shape_type{
      {static_cast<unsigned long>(cells_x())}});
  for(int i = 0; i < cells_x(); i++) {
    deriv(i) = (cv_average(i, -1) - cv_average(i, 0)) / dy();
  }
  return deriv;
}

xt::xtensor<real, 1> PoissonFVMGSolverBase::top_bndry_deriv() const noexcept {
  xt::xtensor<real, 1> deriv(xt::xtensor<real, 1>::shape_type{
      {static_cast<unsigned long>(cells_x())}});
  for(int i = 0; i < cells_x(); i++) {
    deriv(i) = (cv_average(i, cells_y()) - cv_average(i, cells_y() - 1)) / dy();
  }
  return deriv;
}

xt::xtensor<real, 1> PoissonFVMGSolverBase::left_bndry_val() const noexcept {
  constexpr unsigned int quad_pts = 3;
  xt::xtensor<real, 1> avg_val(xt::xtensor<real, 1>::shape_type{
      {static_cast<unsigned long>(cells_y())}});
  constexpr std::array<real, quad_pts> weights{
      {5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0}};
  const std::array<real, quad_pts> offsets{
      {-dy() * std::sqrt(3.0 / 20.0), 0.0, dy() * std::sqrt(3.0 / 20.0)}};
  const auto [_, bc] = bc_.left_bc();
  const real x       = left_x(0);
  for(int j = 0; j < cells_y(); j++) {
    const real y = median_y(j);
    avg_val(j)   = 0.0;
    for(unsigned int k = 0; k < quad_pts; k++) {
      avg_val(j) += bc(x, y + offsets[k]) * weights[k];
    }
    avg_val(j) *= dy();
  }
  return avg_val;
}

xt::xtensor<real, 1> PoissonFVMGSolverBase::right_bndry_val() const noexcept {
  constexpr unsigned int quad_pts = 3;
  xt::xtensor<real, 1> avg_val(xt::xtensor<real, 1>::shape_type{
      {static_cast<unsigned long>(cells_y())}});
  constexpr std::array<real, quad_pts> weights{
      {5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0}};
  const std::array<real, quad_pts> offsets{
      {-dy() * std::sqrt(3.0 / 20.0), 0.0, dy() * std::sqrt(3.0 / 20.0)}};
  const auto [_, bc] = bc_.right_bc();
  const real x       = left_x(cells_x());
  for(int j = 0; j < cells_y(); j++) {
    const real y = median_y(j);
    avg_val(j)   = 0.0;
    for(unsigned int k = 0; k < quad_pts; k++) {
      avg_val(j) += bc(x, y + offsets[k]) * weights[k];
    }
    avg_val(j) *= dy();
  }
  return avg_val;
}

xt::xtensor<real, 1> PoissonFVMGSolverBase::bottom_bndry_val() const noexcept {
  constexpr unsigned int quad_pts = 3;
  xt::xtensor<real, 1> avg_val(xt::xtensor<real, 1>::shape_type{
      {static_cast<unsigned long>(cells_y())}});
  constexpr std::array<real, quad_pts> weights{
      {5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0}};
  const std::array<real, quad_pts> offsets{
      {-dx() * std::sqrt(3.0 / 20.0), 0.0, dx() * std::sqrt(3.0 / 20.0)}};
  const auto [_, bc] = bc_.bottom_bc();
  const real y       = bottom_y(0);
  for(int i = 0; i < cells_x(); i++) {
    const real x = median_x(i);
    avg_val(i)   = 0.0;
    for(unsigned int k = 0; k < quad_pts; k++) {
      avg_val(i) += bc(x + offsets[k], y) * weights[k];
    }
    avg_val(i) *= dx();
  }
  return avg_val;
}

xt::xtensor<real, 1> PoissonFVMGSolverBase::top_bndry_val() const noexcept {
  constexpr unsigned int quad_pts = 3;
  xt::xtensor<real, 1> avg_val(xt::xtensor<real, 1>::shape_type{
      {static_cast<unsigned long>(cells_y())}});
  constexpr std::array<real, quad_pts> weights{
      {5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0}};
  const std::array<real, quad_pts> offsets{
      {-dx() * std::sqrt(3.0 / 20.0), 0.0, dx() * std::sqrt(3.0 / 20.0)}};
  const auto [_, bc] = bc_.top_bc();
  const real y       = bottom_y(cells_y());
  for(int i = 0; i < cells_x(); i++) {
    const real x = median_x(i);
    avg_val(i)   = 0.0;
    for(unsigned int k = 0; k < quad_pts; k++) {
      avg_val(i) += bc(x + offsets[k], y) * weights[k];
    }
    avg_val(i) *= dx();
  }
  return avg_val;
}

BoundaryConditions mg_bc(const BoundaryConditions &src, const real coarse_dx,
                         const real coarse_dy) {
  return BoundaryConditions(
      src.left_bc().first, zero_source, src.right_bc().first, zero_source,
      src.bottom_bc().first, zero_source, src.top_bc().first, zero_source);
}

template <int mg_levels_>
PoissonFVMGSolver<mg_levels_>::PoissonFVMGSolver(
    const std::pair<real, real> &corner_1,
    const std::pair<real, real> &corner_2, const size_t cells_x,
    const size_t cells_y) noexcept
    : PoissonFVMGSolver<mg_levels_>(corner_1, corner_2, cells_x, cells_y,
                                    zero_source) {}

template <int mg_levels_>
PoissonFVMGSolver<mg_levels_>::PoissonFVMGSolver(
    const std::pair<real, real> &corner_1,
    const std::pair<real, real> &corner_2, const size_t cells_x,
    const size_t cells_y, const BoundaryConditions &bc) noexcept
    : PoissonFVMGSolver<mg_levels_>(corner_1, corner_2, cells_x, cells_y, bc,
                                    zero_source) {}

template <int mg_levels_>
PoissonFVMGSolver<mg_levels_>::PoissonFVMGSolver(
    const std::pair<real, real> &corner_1,
    const std::pair<real, real> &corner_2, const size_t cells_x,
    const size_t cells_y,
    const std::function<real(real, real)> &source) noexcept
    : PoissonFVMGSolver<mg_levels_>(corner_1, corner_2, cells_x, cells_y,
                                    BoundaryConditions(), source) {}

template <int mg_levels_>
PoissonFVMGSolver<mg_levels_>::PoissonFVMGSolver(
    const std::pair<real, real> &corner_1,
    const std::pair<real, real> &corner_2, const size_t cells_x,
    const size_t cells_y, const BoundaryConditions &bc,
    const std::function<real(real, real)> &source) noexcept
    : PoissonFVMGSolverBase(corner_1, corner_2, cells_x, cells_y, bc, source),
      multilev_(corner_1, corner_2, cells_x / 2, cells_y / 2,
                mg_bc(bc, this->dx() * 2.0, this->dy() * 2.0), zero_source) {
#ifndef NDEBUG
  assert(cells_x % (1 << (mg_levels_ - 1)) == 0);
  assert(cells_y % (1 << (mg_levels_ - 1)) == 0);
  assert(multilev_.bc_.bottom_bc().first == this->bc_.bottom_bc().first);
  assert(multilev_.bc_.top_bc().first == this->bc_.top_bc().first);
  for(size_t i = 0; i < cells_x; i++) {
    const real bottom_y = this->bottom_y(0);
    const real x        = this->median_x(i);
    assert(multilev_.bc_.bottom_bc().second(x, bottom_y) == 0.0);
    const real top_y = this->top_y(cells_y);
    assert(multilev_.bc_.top_bc().second(x, top_y) == 0.0);
  }
  assert(multilev_.bc_.left_bc().first == this->bc_.left_bc().first);
  assert(multilev_.bc_.right_bc().first == this->bc_.right_bc().first);
  for(size_t j = 0; j < cells_y; j++) {
    const real left_x = this->left_x(0);
    const real y      = this->median_y(j);
    assert(multilev_.bc_.left_bc().second(left_x, y) == 0.0);
    const real right_x = this->right_x(cells_x);
    assert(multilev_.bc_.right_bc().second(right_x, y) == 0.0);
  }
#endif  // NDEBUG
}

PoissonFVMGSolver<1>::PoissonFVMGSolver(const std::pair<real, real> &corner_1,
                                        const std::pair<real, real> &corner_2,
                                        const size_t cells_x,
                                        const size_t cells_y) noexcept
    : PoissonFVMGSolver<1>(corner_1, corner_2, cells_x, cells_y, zero_source) {}

PoissonFVMGSolver<1>::PoissonFVMGSolver(const std::pair<real, real> &corner_1,
                                        const std::pair<real, real> &corner_2,
                                        const size_t cells_x,
                                        const size_t cells_y,
                                        const BoundaryConditions &bc) noexcept
    : PoissonFVMGSolver<1>(corner_1, corner_2, cells_x, cells_y, bc,
                           zero_source) {}

PoissonFVMGSolver<1>::PoissonFVMGSolver(
    const std::pair<double, double> &corner_1,
    const std::pair<double, double> &corner_2, size_t cells_x, size_t cells_y,
    const std::function<double(double, double)> &source) noexcept
    : PoissonFVMGSolver<1>(corner_1, corner_2, cells_x, cells_y,
                           BoundaryConditions(), source) {}

PoissonFVMGSolver<1>::PoissonFVMGSolver(
    const std::pair<double, double> &corner_1,
    const std::pair<double, double> &corner_2, size_t cells_x, size_t cells_y,
    const BoundaryConditions &bc,
    const std::function<double(double, double)> &source) noexcept
    : PoissonFVMGSolverBase(corner_1, corner_2, cells_x, cells_y, bc, source) {}

// Explicitly instantiate multilevel implementations up to 10 levels
template real PoissonFVMGSolverBase::poisson_pgs_or<2>(const real);
template real PoissonFVMGSolverBase::poisson_pgs_or<4>(const real);
template class PoissonFVMGSolver<2>;
template class PoissonFVMGSolver<3>;
template class PoissonFVMGSolver<4>;
template class PoissonFVMGSolver<5>;
template class PoissonFVMGSolver<6>;
template class PoissonFVMGSolver<7>;
template class PoissonFVMGSolver<8>;
template class PoissonFVMGSolver<9>;
template class PoissonFVMGSolver<10>;
