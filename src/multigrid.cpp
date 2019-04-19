
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
  for(int i = -ghost_cells(); i < static_cast<int>(this->cells_x()) + ghost_cells();
      i++) {
    for(int j = -ghost_cells(); j < static_cast<int>(this->cells_y()) + ghost_cells();
        j++) {
      cv_average(i, j) = f(median_x(i), median_y(j));
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
    const real dx, const real dy, const BC_Type left_t,
    const std::function<real(real, real)> &left_bc, const BC_Type right_t,
    const std::function<real(real, real)> &right_bc, const BC_Type top_t,
    const std::function<real(real, real)> &top_bc, const BC_Type bottom_t,
    const std::function<real(real, real)> &bottom_bc)
    : left_t_(left_t),
      right_t_(right_t),
      top_t_(top_t),
      bottom_t_(bottom_t),
      left_bc_(left_bc),
      right_bc_(right_bc),
      top_bc_(top_bc),
      bottom_bc_(bottom_bc),
      horiz_basis(),
      vert_basis() {
  construct_bases(dx, dy);
}

void BoundaryConditions::construct_bases(const real dx, const real dy) {
  // Our system of equations in matrix form is
  // [[0         0           0       1]
  //  [0         dx^3/12     0       dx]
  //  [5/4 dx^4  13/12 dx^3  dx^2    dx]
  //  [17/2 dx^4  49/12 dx^3  2 dx^2  dx]]
  // Our basis is given by inverting this matrix, which is
  // 1/24 [[96/dx^3    -92/dx^4   -8/dx^4    4/dx^4],
  //       [-288/dx^2   288/dx^3   0         0],
  //       [168/dx     -197/dx^2   34/dx^2  -5/dx^2
  //       [24          0          0         0]]
  // Each column is one of our polynomials coefficients,
  // starting at the top with the highest exponent,
  // ending at the bottom with the smallest exponent
  auto basis_fill = [](std::array<Numerical::Polynomial<real, 3, 1>, 4> &basis,
                       const real diff) {
    basis[0].coeff({3}) = 96.0 / (diff * diff * diff * 24.0);
    basis[0].coeff({2}) = -288.0 / (diff * diff * 24.0);
    basis[0].coeff({1}) = 168.0 / (diff * 24.0);
    basis[0].coeff({0}) = 1.0;

    basis[1].coeff({3}) = -92.0 / (diff * diff * diff * diff * 24.0);
    basis[1].coeff({2}) = 288.0 / (diff * diff * diff * 24.0);
    basis[1].coeff({1}) = -197.0 / (diff * diff * 24.0);
    basis[1].coeff({0}) = 0.0;

    basis[2].coeff({3}) = -8.0 / (diff * diff * diff * diff * 24.0);
    basis[2].coeff({2}) = 0.0;
    basis[2].coeff({1}) = -34.0 / (diff * diff * 24.0);
    basis[2].coeff({0}) = 0.0;

    basis[3].coeff({3}) = 4.0 / (diff * diff * diff * diff * 24.0);
    basis[3].coeff({2}) = 0.0;
    basis[3].coeff({1}) = -5.0 / (diff * diff * 24.0);
    basis[3].coeff({0}) = 0.0;
  };
  basis_fill(horiz_basis, dx);
  basis_fill(vert_basis, dy);
}

void BoundaryConditions::apply(Mesh &mesh) const noexcept {
  // Uses a cubic interpolating polynomial with no cross terms to interpolate
  // the values of the ghost cells, giving a 4th order implementation of the
  // boundary conditions
  //
  // Currently only implemented for Dirichlet boundary conditions
  assert(bottom_t_ == BC_Type::dirichlet);
  assert(top_t_ == BC_Type::dirichlet);
  assert(left_t_ == BC_Type::dirichlet);
  assert(right_t_ == BC_Type::dirichlet);
  // Set the top and bottom ghost cells
  for(int i = 0; i < mesh.cells_x(); i++) {
    const real x        = mesh.median_x(i);
    const real y_bottom = mesh.bottom_y(0);
    const real y_top    = mesh.bottom_y(mesh.cells_y());
    const auto bottom_poly =
        vert_basis[0] * bottom_bc_(x, y_bottom) + vert_basis[1] * mesh[{i, 0}] +
        vert_basis[2] * mesh[{i, 1}] + vert_basis[3] * mesh[{i, 2}];
    mesh[{i, -2}]       = bottom_poly.eval(mesh.median_y(-2));
    mesh[{i, -1}]       = bottom_poly.eval(mesh.median_y(-1));
    const auto top_poly = vert_basis[0] * top_bc_(x, y_top) +
                          vert_basis[1] * mesh[{i, mesh.cells_y() - 1}] +
                          vert_basis[2] * mesh[{i, mesh.cells_y() - 2}] +
                          vert_basis[3] * mesh[{i, mesh.cells_y() - 3}];
    mesh[{i, mesh.cells_y()}]     = top_poly.eval(mesh.median_y(-2));
    mesh[{i, mesh.cells_y() + 1}] = top_poly.eval(mesh.median_y(-1));
  }
  for(int j = 0; j < mesh.cells_y(); j++) {
    const real y       = mesh.median_y(j);
    const real x_left  = mesh.left_x(0);
    const real x_right = mesh.left_x(mesh.cells_x());
    const auto left_poly =
        horiz_basis[0] * left_bc_(x_left, y) + horiz_basis[1] * mesh[{0, j}] +
        horiz_basis[2] * mesh[{1, j}] + horiz_basis[3] * mesh[{2, j}];
    mesh[{-2, j}]         = left_poly.eval(mesh.median_x(-2));
    mesh[{-1, j}]         = left_poly.eval(mesh.median_x(-1));
    const auto right_poly = horiz_basis[0] * right_bc_(x_right, y) +
                            horiz_basis[1] * mesh[{mesh.cells_x() - 1, j}] +
                            horiz_basis[2] * mesh[{mesh.cells_x() - 2, j}] +
                            horiz_basis[3] * mesh[{mesh.cells_x() - 3, j}];
    mesh[{mesh.cells_x(), j}]     = right_poly.eval(mesh.median_x(-1));
    mesh[{mesh.cells_x() + 1, j}] = right_poly.eval(mesh.median_x(-2));
  }
  // Prolongating with bilinear interpolation requires the corner ghost cell to
  // be a reasonable approximation of its actual value
  // - set it to the value of the internal corner cell.
  // Better might be using the inner corner cell gradient to estimate it
  mesh[{-1, -1}]             = mesh[{0, 0}];
  mesh[{-1, mesh.cells_y()}] = mesh[{0, mesh.cells_y() - 1}];
  mesh[{mesh.cells_x(), -1}] = mesh[{mesh.cells_x() - 1, 0}];
  mesh[{mesh.cells_x(), mesh.cells_y()}] =
      mesh[{mesh.cells_x() - 1, mesh.cells_y() - 1}];
}

// Use this to enable initializing the homogeneous system easy
constexpr real zero_source(real, real) noexcept { return 0.0; }

template <unsigned int order_>
PoissonFVMGSolverBase<order_>::PoissonFVMGSolverBase(
    const std::pair<real, real> &corner_1,
    const std::pair<real, real> &corner_2, const size_t cells_x,
    const size_t cells_y) noexcept
    : PoissonFVMGSolverBase<order_>(corner_1, corner_2, cells_x, cells_y,
                                    zero_source) {}

template <unsigned int order_>
PoissonFVMGSolverBase<order_>::PoissonFVMGSolverBase(
    const std::pair<real, real> &corner_1,
    const std::pair<real, real> &corner_2, const size_t cells_x,
    const size_t cells_y, const BoundaryConditions &bc) noexcept
    : PoissonFVMGSolverBase<order_>(corner_1, corner_2, cells_x, cells_y, bc,
                                    zero_source) {}

template <unsigned int order_>
PoissonFVMGSolverBase<order_>::PoissonFVMGSolverBase(
    const std::pair<real, real> &corner_1,
    const std::pair<real, real> &corner_2, const size_t cells_x,
    const size_t cells_y,
    const std::function<real(real, real)> &source) noexcept
    : PoissonFVMGSolverBase<order_>(
          corner_1, corner_2, cells_x, cells_y,
          BoundaryConditions(
              std::abs(corner_1.first - corner_2.first) / cells_x,
              std::abs(corner_1.second - corner_2.second) / cells_y),
          source) {}

template <unsigned int order_>
PoissonFVMGSolverBase<order_>::PoissonFVMGSolverBase(
    const std::pair<real, real> &corner_1,
    const std::pair<real, real> &corner_2, const size_t cells_x,
    const size_t cells_y, const BoundaryConditions &bc,
    const std::function<real(real, real)> &source) noexcept
    : Mesh(corner_1, corner_2, cells_x, cells_y, 2),
      bc_(bc),
      source_(corner_1, corner_2, cells_x, cells_y, source, 0) {}

template <>
real PoissonFVMGSolverBase<2>::delta(const int i, const int j) const noexcept {
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
real PoissonFVMGSolverBase<4>::delta(const int i, const int j) const noexcept {
  // Computes the difference between the Laplacian and the source term
  assert(i >= 0);
  assert(i < cells_x());
  assert(j >= 0);
  assert(j < cells_y());
  // Our problem is the form of \del u = f
  // The residual is then just r = \del u - f
  return ((-cv_average(i - 2, j) + 16.0 * cv_average(i - 1, j) -
           30.0 * cv_average(i, j) + 16.0 * cv_average(i + 1, j) -
           cv_average(i + 2, j)) /
              (12.0 * dx_ * dx_) +
          (cv_average(i, j - 2) + 16.0 * cv_average(i, j - 1) -
           30.0 * cv_average(i, j) + 16.0 * cv_average(i, j + 1) -
           cv_average(i, j + 2)) /
              (12.0 * dy_ * dy_)) -
         source_[{i, j}];
}

template <unsigned int order_>
real PoissonFVMGSolverBase<order_>::poisson_pgs_or(
    const real or_term) noexcept {
  real max_diff = -std::numeric_limits<real>::infinity();
  bc_.apply(*this);
  const real diff_scale =
      (dx_ * dx_ * dy_ * dy_ / (2.0 * (dx_ * dx_ + dy_ * dy_)));
  for(int i = 0; i < cells_x(); i++) {
    for(int j = 0; j < cells_y(); j++) {
      const real diff = or_term * diff_scale * delta(i, j);
      max_diff        = std::max(max_diff, std::abs(diff));
      cv_average(i, j) += diff;
    }
  }
  return max_diff;
}

template <unsigned int order_>
void PoissonFVMGSolverBase<order_>::restrict(
    const PoissonFVMGSolverBase &src) noexcept {
  // We're assuming the cells are uniform
  assert(src.cells_x() == 2 * cells_x());
  assert(src.cells_y() == 2 * cells_y());

  for(int i = 0; i < cells_x(); i++) {
    for(int j = 0; j < cells_y(); j++) {
      // The restriction sets the source to the average of the residual for the
      // cells in the same area
      source_[{i, j}] =
          -0.25 *
          (src.delta(2 * i, 2 * j) + src.delta(2 * i, 2 * j + 1) +
           src.delta(2 * i + 1, 2 * j) + src.delta(2 * i + 1, 2 * j + 1));
      // It's not clear how to best precondition the coarser grids, so just
      // set it to 0, it appears to work reasonably well
      cv_average(i, j) = 0.0;
    }
  }
}

template <unsigned int order_>
real PoissonFVMGSolverBase<order_>::prolongate(Mesh &dest) const noexcept {
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

BoundaryConditions mg_bc(const BoundaryConditions &src, const real coarse_dx,
                         const real coarse_dy) {
  return BoundaryConditions(coarse_dx, coarse_dy, src.left_bc().first,
                            zero_source, src.right_bc().first, zero_source,
                            src.bottom_bc().first, zero_source,
                            src.top_bc().first, zero_source);
}

template <int mg_levels_, unsigned int order_>
PoissonFVMGSolver<mg_levels_, order_>::PoissonFVMGSolver(
    const std::pair<real, real> &corner_1,
    const std::pair<real, real> &corner_2, const size_t cells_x,
    const size_t cells_y) noexcept
    : PoissonFVMGSolver<mg_levels_, order_>(corner_1, corner_2, cells_x,
                                            cells_y, zero_source) {}

template <int mg_levels_, unsigned int order_>
PoissonFVMGSolver<mg_levels_, order_>::PoissonFVMGSolver(
    const std::pair<real, real> &corner_1,
    const std::pair<real, real> &corner_2, const size_t cells_x,
    const size_t cells_y, const BoundaryConditions &bc) noexcept
    : PoissonFVMGSolver<mg_levels_, order_>(corner_1, corner_2, cells_x,
                                            cells_y, bc, zero_source) {}

template <int mg_levels_, unsigned int order_>
PoissonFVMGSolver<mg_levels_, order_>::PoissonFVMGSolver(
    const std::pair<real, real> &corner_1,
    const std::pair<real, real> &corner_2, const size_t cells_x,
    const size_t cells_y,
    const std::function<real(real, real)> &source) noexcept
    : PoissonFVMGSolver<mg_levels_, order_>(
          corner_1, corner_2, cells_x, cells_y,
          BoundaryConditions(
              std::abs(corner_1.first - corner_2.first) / cells_x,
              std::abs(corner_1.second - corner_2.second) / cells_y),
          source) {}

template <int mg_levels_, unsigned int order_>
PoissonFVMGSolver<mg_levels_, order_>::PoissonFVMGSolver(
    const std::pair<real, real> &corner_1,
    const std::pair<real, real> &corner_2, const size_t cells_x,
    const size_t cells_y, const BoundaryConditions &bc,
    const std::function<real(real, real)> &source) noexcept
    : PoissonFVMGSolverBase<order_>(corner_1, corner_2, cells_x, cells_y, bc,
                                    source),
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

template <unsigned int order_>
PoissonFVMGSolver<1, order_>::PoissonFVMGSolver(
    const std::pair<real, real> &corner_1,
    const std::pair<real, real> &corner_2, const size_t cells_x,
    const size_t cells_y) noexcept
    : PoissonFVMGSolver<1, order_>(corner_1, corner_2, cells_x, cells_y,
                                   zero_source) {}

template <unsigned int order_>
PoissonFVMGSolver<1, order_>::PoissonFVMGSolver(
    const std::pair<real, real> &corner_1,
    const std::pair<real, real> &corner_2, const size_t cells_x,
    const size_t cells_y, const BoundaryConditions &bc) noexcept
    : PoissonFVMGSolver<1, order_>(corner_1, corner_2, cells_x, cells_y, bc,
                                   zero_source) {}

template <unsigned int order_>
PoissonFVMGSolver<1, order_>::PoissonFVMGSolver(
    const std::pair<double, double> &corner_1,
    const std::pair<double, double> &corner_2, size_t cells_x, size_t cells_y,
    const std::function<double(double, double)> &source) noexcept
    : PoissonFVMGSolver<1, order_>(
          corner_1, corner_2, cells_x, cells_y,
          BoundaryConditions(
              std::abs(corner_1.first - corner_2.first) / cells_x,
              std::abs(corner_1.second - corner_2.second) / cells_y),
          source) {}

template <unsigned int order_>
PoissonFVMGSolver<1, order_>::PoissonFVMGSolver(
    const std::pair<double, double> &corner_1,
    const std::pair<double, double> &corner_2, size_t cells_x, size_t cells_y,
    const BoundaryConditions &bc,
    const std::function<double(double, double)> &source) noexcept
    : PoissonFVMGSolverBase<order_>(corner_1, corner_2, cells_x, cells_y, bc,
                                    source) {}

// Explicitly instantiate the Poisson solver implementations
template class PoissonFVMGSolverBase<2>;
template class PoissonFVMGSolverBase<4>;
// Explicitly instantiate multilevel implementations up to 10 levels
template class PoissonFVMGSolver<1, 2>;
template class PoissonFVMGSolver<2, 2>;
template class PoissonFVMGSolver<3, 2>;
template class PoissonFVMGSolver<4, 2>;
template class PoissonFVMGSolver<5, 2>;
template class PoissonFVMGSolver<6, 2>;
template class PoissonFVMGSolver<7, 2>;
template class PoissonFVMGSolver<8, 2>;
template class PoissonFVMGSolver<9, 2>;
template class PoissonFVMGSolver<10, 2>;

template class PoissonFVMGSolver<1, 4>;
template class PoissonFVMGSolver<2, 4>;
template class PoissonFVMGSolver<3, 4>;
template class PoissonFVMGSolver<4, 4>;
template class PoissonFVMGSolver<5, 4>;
template class PoissonFVMGSolver<6, 4>;
template class PoissonFVMGSolver<7, 4>;
template class PoissonFVMGSolver<8, 4>;
template class PoissonFVMGSolver<9, 4>;
template class PoissonFVMGSolver<10, 4>;
