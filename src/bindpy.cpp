
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <sstream>
#include <vector>

#include "multigrid.hpp"

namespace py = pybind11;

// This just exports all the objects needed in Python

template <int mg_levels>
void define_solver(py::module module) {
  std::stringstream ss;
  ss << "PoissonFVMG_" << mg_levels;
  using Solver = PoissonFVMGSolver<mg_levels>;
  py::class_<Solver, PoissonFVMGSolverBase> poisson(module, ss.str().c_str());
  poisson
      .def(
          py::init<const std::pair<real, real> &, const std::pair<real, real> &,
                   size_t, size_t, const BoundaryConditions &,
                   const std::function<real(real, real)> &>())
      .def(
          py::init<const std::pair<real, real> &, const std::pair<real, real> &,
                   size_t, size_t, const std::function<real(real, real)> &>())
      .def(
          py::init<const std::pair<real, real> &, const std::pair<real, real> &,
                   size_t, size_t, const BoundaryConditions &>())
      .def(py::init<const std::pair<real, real> &,
                    const std::pair<real, real> &, size_t, size_t>());
  using container = std::vector<std::tuple<int, int, real>>;
  poisson.def("solve_2", &Solver::template solve<2, container>);
  poisson.def("solve_4", &Solver::template solve<4, container>);
  if constexpr(mg_levels > 1) {
    poisson.def("error_solver", &Solver::error_mesh);
    define_solver<mg_levels - 1>(module);
  }
}

PYBIND11_MODULE(multigrid, module) {
  module.doc() = "C++ Multigrid Solver for the Poisson Equation";
  py::class_<Mesh> mesh(module, "Mesh");
  mesh.def(py::init<std::pair<real, real>, std::pair<real, real>, size_t,
                    size_t, std::function<real(real, real)>>())
      .def(py::init<std::pair<real, real>, std::pair<real, real>, size_t,
                    size_t>());
  mesh.def("__getitem__",
           [](const Mesh &m, std::pair<int, int> p) { return m[p]; })
      .def("__setitem__",
           [](Mesh &m, std::pair<int, int> p, real val) {
             m[p] = val;
             return m;
           })
      .def("cells_x", &Mesh::cells_x)
      .def("cells_y", &Mesh::cells_y)
      .def("median_x", &Mesh::median_x)
      .def("median_y", &Mesh::median_y)
      .def("dx", &Mesh::dx)
      .def("dy", &Mesh::dy)
      .def("array",
           [](Mesh &m) {
             return py::array((m.cells_x() + 2 * m.ghost_cells()) *
                                  (m.cells_y() + 2 * m.ghost_cells()),
                              m.data())
                 .attr("reshape")(m.cells_x() + 2 * m.ghost_cells(),
                                  m.cells_y() + 2 * m.ghost_cells());
           })
      .def("interpolate", &Mesh::interpolate)
      .def("grid_x",
           [](const Mesh &m) {
             xt::xtensor<real, 2> grid(std::array<size_t, 2>{
                 static_cast<size_t>(m.cells_x() + 2 * m.ghost_cells()),
                 static_cast<size_t>(m.cells_y() + 2 * m.ghost_cells())});
             for(int i = 0; i < m.cells_x() + 2 * m.ghost_cells(); i++) {
               for(int j = 0; j < m.cells_y() + 2 * m.ghost_cells(); j++) {
                 grid(i, j) = m.median_x(i - m.ghost_cells());
               }
             }
             return py::array((m.cells_x() + 2 * m.ghost_cells()) *
                                  (m.cells_y() + 2 * m.ghost_cells()),
                              grid.data())
                 .attr("reshape")(m.cells_x() + 2 * m.ghost_cells(),
                                  m.cells_y() + 2 * m.ghost_cells());
           })
      .def("grid_y", [](const Mesh &m) {
        xt::xtensor<real, 2> grid(std::array<size_t, 2>{
            static_cast<size_t>(m.cells_x() + 2 * m.ghost_cells()),
            static_cast<size_t>(m.cells_y() + 2 * m.ghost_cells())});
        for(int i = 0; i < m.cells_x() + 2 * m.ghost_cells(); i++) {
          for(int j = 0; j < m.cells_y() + 2 * m.ghost_cells(); j++) {
            grid(i, j) = m.median_y(j - m.ghost_cells());
          }
        }
        return py::array((m.cells_x() + 2 * m.ghost_cells()) *
                             (m.cells_y() + 2 * m.ghost_cells()),
                         grid.data())
            .attr("reshape")(m.cells_x() + 2 * m.ghost_cells(),
                             m.cells_y() + 2 * m.ghost_cells());
      });

  py::class_<BoundaryConditions> bc(module, "BoundaryConditions");
  bc.def(py::init<BoundaryConditions::BC_Type, std::function<real(real, real)>,
                  BoundaryConditions::BC_Type, std::function<real(real, real)>,
                  BoundaryConditions::BC_Type, std::function<real(real, real)>,
                  BoundaryConditions::BC_Type,
                  std::function<real(real, real)>>())
      .def(py::init<>())
      .def("left_bc", &BoundaryConditions::left_bc)
      .def("right_bc", &BoundaryConditions::right_bc)
      .def("top_bc", &BoundaryConditions::top_bc)
      .def("bottom_bc", &BoundaryConditions::bottom_bc)
      .def("apply_2", &BoundaryConditions::template apply<2>)
      .def("apply_4", &BoundaryConditions::template apply<4>);

  py::enum_<BoundaryConditions::BC_Type>(bc, "BC_Type")
      .value("dirichlet", BoundaryConditions::BC_Type::dirichlet)
      .value("neumann", BoundaryConditions::BC_Type::neumann)
      .export_values();

  using Base = PoissonFVMGSolverBase;
  py::class_<Base, Mesh> poisson_base(module, "PoissonFVMGBase");
  poisson_base
      .def(
          py::init<const std::pair<real, real> &, const std::pair<real, real> &,
                   size_t, size_t, const BoundaryConditions &,
                   const std::function<real(real, real)> &>())
      .def(
          py::init<const std::pair<real, real> &, const std::pair<real, real> &,
                   size_t, size_t, const std::function<real(real, real)> &>())
      .def(
          py::init<const std::pair<real, real> &, const std::pair<real, real> &,
                   size_t, size_t, const BoundaryConditions &>())
      .def(py::init<const std::pair<real, real> &,
                    const std::pair<real, real> &, size_t, size_t>());
  poisson_base.def("restrict", &Base::restrict)
      .def("prolongate", &Base::prolongate)
      .def("source", &Base::source)
      .def("delta_2", &Base::template delta<2>)
      .def("delta_4", &Base::template delta<4>)
      .def("poisson_pgs_or_2", &Base::template poisson_pgs_or<2>)
      .def("poisson_pgs_or_4", &Base::template poisson_pgs_or<4>)
      .def("apply_bc_2", &Base::template apply_bc<2>)
      .def("apply_bc_4", &Base::template apply_bc<4>)
      .def("delta_grid_2",
           [](const PoissonFVMGSolverBase &m) {
             xt::xtensor<real, 2> grid(
                 std::array<size_t, 2>{static_cast<size_t>(m.cells_x()),
                                       static_cast<size_t>(m.cells_y())});
             for(int i = 0; i < m.cells_x(); i++) {
               for(int j = 0; j < m.cells_y(); j++) {
                 grid(i, j) = m.delta<2>(i, j);
               }
             }
             return py::array(m.cells_x() * m.cells_y(), grid.data())
                 .attr("reshape")(m.cells_x(), m.cells_y());
           })
      .def("delta_grid_4",
           [](const PoissonFVMGSolverBase &m) {
             xt::xtensor<real, 2> grid(
                 std::array<size_t, 2>{static_cast<size_t>(m.cells_x()),
                                       static_cast<size_t>(m.cells_y())});
             for(int i = 0; i < m.cells_x(); i++) {
               for(int j = 0; j < m.cells_y(); j++) {
                 grid(i, j) = m.delta<4>(i, j);
               }
             }
             return py::array(m.cells_x() * m.cells_y(), grid.data())
                 .attr("reshape")(m.cells_x(), m.cells_y());
           })
      .def("left_bndry_deriv",
           [](const Base &m) {
             return py::array(m.cells_y(), m.left_bndry_deriv().data());
           })
      .def("right_bndry_deriv",
           [](const Base &m) {
             return py::array(m.cells_y(), m.right_bndry_deriv().data());
           })
      .def("bottom_bndry_deriv",
           [](const Base &m) {
             return py::array(m.cells_x(), m.bottom_bndry_deriv().data());
           })
      .def("top_bndry_deriv",
           [](const Base &m) {
             return py::array(m.cells_x(), m.top_bndry_deriv().data());
           })
      .def("left_bndry_val",
           [](const Base &m) {
             const xt::xtensor<real, 2> t = m.left_bndry_val();
             return py::array(t.size(), t.data())
                 .attr("reshape")(t.shape()[0], t.shape()[1]);
           })
      .def("right_bndry_val",
           [](const Base &m) {
             const xt::xtensor<real, 2> t = m.right_bndry_val();
             return py::array(t.size(), t.data())
                 .attr("reshape")(t.shape()[0], t.shape()[1]);
           })
      .def("bottom_bndry_val",
           [](const Base &m) {
             const xt::xtensor<real, 2> t = m.bottom_bndry_val();
             return py::array(t.size(), t.data())
                 .attr("reshape")(t.shape()[0], t.shape()[1]);
           })
      .def("top_bndry_val", [](const Base &m) {
        const xt::xtensor<real, 2> t = m.top_bndry_val();
        return py::array(t.size(), t.data())
            .attr("reshape")(t.shape()[0], t.shape()[1]);
      });

  define_solver<10>(module);
}
