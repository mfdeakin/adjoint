
#include <gtest/gtest.h>
#include <cmath>
#include <random>

#include "multigrid.hpp"
#include "polynomial/polynomial.hpp"

constexpr real pi = 3.1415926535897932;

real test_func(real x, real y) { return std::sin(pi * x) * std::sin(pi * y); }
real test_residual(real x, real y) { return -2.0 * pi * pi * test_func(x, y); }

TEST(mesh, cell_overlap) {
  // Verifies that the multigrid cells are lined up with the finer cells
  constexpr int cells_x = 32, cells_y = 32;
  PoissonFVMGSolver<2> fine({0.0, 0.0}, {1.0, 1.0}, cells_x, cells_y);
  const PoissonFVMGSolver<1> &coarse = fine.error_mesh();
  EXPECT_EQ(2 * coarse.cells_x(), fine.cells_x());
  for(int i = 0; i < coarse.cells_x(); i++) {
    EXPECT_NEAR(
        coarse.median_x(i), fine.right_x(i * 2),
        4.0 * std::numeric_limits<real>::epsilon() * coarse.median_x(i));
    EXPECT_NEAR(
        coarse.median_x(i), fine.left_x(i * 2 + 1),
        4.0 * std::numeric_limits<real>::epsilon() * coarse.median_x(i));
    EXPECT_NEAR(coarse.left_x(i), fine.left_x(i * 2),
                4.0 * std::numeric_limits<real>::epsilon() * coarse.left_x(i));
    EXPECT_NEAR(coarse.right_x(i), fine.right_x(i * 2 + 1),
                4.0 * std::numeric_limits<real>::epsilon() * coarse.right_x(i));
  }
  EXPECT_EQ(2 * coarse.cells_y(), fine.cells_y());
  for(int j = 0; j < coarse.cells_y(); j++) {
    EXPECT_NEAR(
        coarse.median_y(j), fine.top_y(j * 2),
        4.0 * std::numeric_limits<real>::epsilon() * coarse.median_y(j));
    EXPECT_NEAR(
        coarse.median_y(j), fine.bottom_y(j * 2 + 1),
        4.0 * std::numeric_limits<real>::epsilon() * coarse.median_y(j));
    EXPECT_NEAR(
        coarse.bottom_y(j), fine.bottom_y(j * 2),
        4.0 * std::numeric_limits<real>::epsilon() * coarse.bottom_y(j));
    EXPECT_NEAR(coarse.top_y(j), fine.top_y(j * 2 + 1),
                4.0 * std::numeric_limits<real>::epsilon() * coarse.top_y(j));
  }
}

TEST(boundary_cond, homogeneous) {
  // Verifies that the boundary conditions are only applied to the ghost cells,
  // and that the boundary value is 0.0
  // Use a cubic polynomial to specify the boundary values
  std::random_device rd;
  std::mt19937_64 rng(rd());
  std::uniform_real_distribution<real> pdf(-1.0, 1.0);
  constexpr int degree = 3;
  Numerical::Polynomial<real, degree, 2> actual;
  // Start with a random polynomial
  actual.coeff_iterator([&actual, &pdf, &rng](const Array<int, 2> &exponents) {
    if(exponents[0] != 0 && exponents[1] != 0) {
      return;
    } else {
      actual.coeff(exponents) = pdf(rng);
    }
  });
  const auto cva_vals = actual.integrate(0).integrate(1);
  // Now construct the mesh and boundary conditions
  constexpr int cells_x = 32, cells_y = 32;
  Mesh fine({0.0, 0.0}, {1.0, 1.0}, cells_x, cells_y, 2);
  auto bndry_val = [actual](const real x, const real y) {
    return actual.eval(x, y);
  };
  for(int i = 0; i < fine.cells_x(); i++) {
    for(int j = 0; j < fine.cells_y(); j++) {
      fine[{i, j}] = (cva_vals.eval(fine.right_x(i), fine.top_y(j)) -
                      cva_vals.eval(fine.left_x(i), fine.bottom_y(j))) /
                     (fine.dx() * fine.dy());
    }
  }
  const Mesh saved(fine);
  // Verify all the cells are correct at the start
  for(int i = 0; i < fine.cells_x(); i++) {
    for(int j = 0; j < fine.cells_y(); j++) {
      EXPECT_EQ((fine[{i, j}]), (saved[{i, j}]));
    }
  }
  const BoundaryConditions bc(BoundaryConditions::BC_Type::dirichlet, bndry_val,
                              BoundaryConditions::BC_Type::dirichlet, bndry_val,
                              BoundaryConditions::BC_Type::dirichlet, bndry_val,
                              BoundaryConditions::BC_Type::dirichlet,
                              bndry_val);
  bc.template apply<4>(fine);
  // Verify the internal cells haven't changed
  for(int i = 0; i < fine.cells_x(); i++) {
    for(int j = 0; j < fine.cells_y(); j++) {
      EXPECT_EQ((fine[{i, j}]), (saved[{i, j}]));
    }
  }
  printf("Cubic Interpolant Quadrature Points:\n");
  constexpr int test_y = 5;
  printf("% .4e, % .4e\n", fine.left_x(0),
         bndry_val(fine.left_x(0), fine.median_y(test_y)));
  printf("% .4e, % .4e\n", fine.median_x(0), fine[{0, test_y}]);
  printf("% .4e, % .4e\n", fine.median_x(1), fine[{1, test_y}]);
  printf("% .4e, % .4e\n", fine.median_x(2), fine[{2, test_y}]);

  printf("% .4e, % .4e\n", fine.median_x(-1), fine[{-1, test_y}]);
  printf("% .4e, % .4e\n", fine.median_x(-2), fine[{-2, test_y}]);
  // Verify the ghost cells match the expected values
  for(int i = 0; i < fine.ghost_cells(); i++) {
    for(int j = 0; j < fine.cells_y(); j++) {
      EXPECT_NEAR((fine[{-1 - i, j}]),
                  (cva_vals.eval(fine.right_x(-1 - i), fine.top_y(j)) -
                   cva_vals.eval(fine.left_x(-1 - i), fine.bottom_y(j))) /
                      (fine.dx() * fine.dy()),
                  1e-10);
      // EXPECT_NEAR(
      //     (fine[{fine.cells_x() + i, j}]),
      //     actual.eval(fine.median_x(fine.cells_x() + i), fine.median_y(j)),
      //     1e-10);
    }
  }
}

TEST(poisson, residual) {
  constexpr int cells_x = 32, cells_y = 32;

  // The solver for the homogeneous Poisson problem
  PoissonFVMGSolverBase test({0.0, 0.0}, {1.0, 1.0}, cells_x, cells_y);
  // Initialize its solution guess with the test function
  for(int i = 0; i < test.cells_x(); i++) {
    const real x = test.median_x(i);
    for(int j = 0; j < test.cells_y(); j++) {
      const real y = test.median_y(j);
      test[{i, j}] = test_func(x, y);
    }
  }
  for(int i = 2; i < test.cells_x() - 2; i++) {
    const real x = test.median_x(i);
    for(int j = 2; j < test.cells_y() - 2; j++) {
      const real y = test.median_y(j);
      EXPECT_NEAR(test.template delta<2>(i, j), test_residual(x, y), 2e-2);
      EXPECT_NEAR(test.template delta<4>(i, j), test_residual(x, y), 5e-5);
    }
  }
}

TEST(multigrid, transfer_simple) {
  constexpr int cells_x = 128, cells_y = 128;

  // The solver for the homogeneous Poisson problem
  PoissonFVMGSolverBase src({0.0, 0.0}, {1.0, 1.0}, cells_x, cells_y,
                            test_func);
  for(int i = -1; i <= src.cells_x(); i++) {
    const real x = src.median_x(i);
    for(int j = -1; j <= src.cells_y(); j++) {
      const real y = src.median_y(j);
      EXPECT_EQ((src[{i, j}]), 0.0);
      if(i > 0 && i < src.cells_x() && j > 0 && j < src.cells_y()) {
        EXPECT_NEAR(src.template delta<2>(i, j), -test_func(x, y), 5e-4);
        EXPECT_NEAR(src.template delta<4>(i, j), -test_func(x, y), 5e-4);
      }
    }
  }
  PoissonFVMGSolverBase dest({0.0, 0.0}, {1.0, 1.0}, cells_x / 2, cells_y / 2);
  dest.restrict(src);
  const Mesh &restricted = dest.source();
  real err_norm          = 0.0;
  for(int i = 0; i < restricted.cells_x(); i++) {
    const real x = restricted.median_x(i);
    for(int j = 0; j < restricted.cells_y(); j++) {
      const real y = restricted.median_y(j);
      dest[{i, j}] = test_func(x, y);

      EXPECT_NEAR((restricted[{i, j}]), test_func(x, y), 7e-4);
      const real err = restricted[{i, j}] - test_func(x, y);
      err_norm += err * err;

      const real x1 = src.median_x(2 * i);
      const real x2 = src.median_x(2 * i + 1);
      const real y1 = src.median_y(2 * j);
      const real y2 = src.median_y(2 * j + 1);
      EXPECT_NEAR((restricted[{i, j}]),
                  0.25 * (test_func(x1, y1) + test_func(x1, y2) +
                          test_func(x2, y1) + test_func(x2, y2)),
                  5e-4);
    }
  }

  err_norm = 0.0;
  Mesh result({0.0, 0.0}, {1.0, 1.0}, cells_x, cells_y);
  dest.prolongate(result);
  for(int i = 1; i < result.cells_x() - 1; i++) {
    const real x = src.median_x(i);
    for(int j = 1; j < result.cells_y() - 1; j++) {
      const real y = src.median_y(j);
      EXPECT_NEAR((result[{i, j}]), test_func(x, y), 2e-3);
      const real err = result[{i, j}] - test_func(x, y);
      err_norm += err * err;
    }
  }
}

TEST(multigrid, transfer_combined) {
  constexpr int cells_x = 64, cells_y = 64;

  // The solver for the homogeneous Poisson problem
  PoissonFVMGSolverBase src({0.0, 0.0}, {1.0, 1.0}, cells_x, cells_y);
  // Initialize its solution guess and ghost cells with the test function
  for(int i = -1; i <= src.cells_x(); i++) {
    const real x = src.median_x(i);
    for(int j = -1; j <= src.cells_y(); j++) {
      const real y = src.median_y(j);
      src[{i, j}]  = test_func(x, y);
    }
  }
  // src.delta(i, j) = \del test_func(x, y)
  PoissonFVMGSolver<1> dest({0.0, 0.0}, {1.0, 1.0}, cells_x / 2, cells_y / 2);
  dest.restrict(src);
  const Mesh &restricted = dest.source();
  for(int i = 0; i < restricted.cells_x(); i++) {
    const real x = restricted.median_x(i);
    for(int j = 0; j < restricted.cells_y(); j++) {
      const real y = restricted.median_y(j);
      dest[{i, j}] = test_func(x, y);
      // Note that this error is the compounded error of the second derivative
      // approximation and the error from the transfer
      // The error is within the same bounds as the second derivative
      // approximation on a mesh of the same size, suggesting it's correct
      EXPECT_NEAR((-restricted[{i, j}]), test_residual(x, y), 2e-2);
    }
  }
  Mesh result({0.0, 0.0}, {1.0, 1.0}, cells_x, cells_y);
  dest.prolongate(result);
  for(int i = 0; i < result.cells_x(); i++) {
    const real x = src.median_x(i);
    for(int j = 0; j < result.cells_y(); j++) {
      const real y = src.median_y(j);
      EXPECT_NEAR((result[{i, j}]), test_func(x, y), 2e-2);
    }
  }
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
