/**
 * bindings.cpp — pybind11 bindings for the single-phase solver.
 *
 * Exposes:
 *   opal_single_phase.SinglePhaseSolver(N, R, C, rho, Cp, V)
 *     .step(p, T, mdot, bc, dt)          → modifies arrays in-place
 *     .solve(p, T, mdot, bc, dt,
 *            n_steps, stride=1)           → np.ndarray shape (n_snap, 2N+1)
 *
 *   opal_single_phase.BoundaryConditions(p_in, p_out, T_in)
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "solver.hpp"

namespace py = pybind11;
using namespace opal;

// ---------------------------------------------------------------------------
// Helpers — numpy ↔ std::vector conversion
// ---------------------------------------------------------------------------

static std::vector<double> to_vec(py::array_t<double> arr) {
    auto buf = arr.request();
    if (buf.ndim != 1)
        throw std::invalid_argument("Expected 1-D array");
    auto* ptr = static_cast<double*>(buf.ptr);
    return std::vector<double>(ptr, ptr + buf.size);
}

static void copy_back(py::array_t<double>& arr, const std::vector<double>& vec) {
    auto buf = arr.request();
    std::memcpy(buf.ptr, vec.data(), vec.size() * sizeof(double));
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

PYBIND11_MODULE(opal_single_phase, m) {
    m.doc() = "OPAL Phase 1 single-phase semi-implicit staggered-mesh solver";

    // BoundaryConditions ---------------------------------------------------
    py::class_<BoundaryConditions>(m, "BoundaryConditions")
        .def(py::init<double, double, double>(),
             py::arg("p_in"), py::arg("p_out"), py::arg("T_in"),
             "Boundary conditions: inlet pressure [Pa], outlet pressure [Pa], "
             "inlet temperature [K]")
        .def_readwrite("p_in",  &BoundaryConditions::p_in)
        .def_readwrite("p_out", &BoundaryConditions::p_out)
        .def_readwrite("T_in",  &BoundaryConditions::T_in)
        .def("__repr__", [](const BoundaryConditions& bc) {
            return "<BoundaryConditions p_in=" + std::to_string(bc.p_in) +
                   " p_out=" + std::to_string(bc.p_out) +
                   " T_in=" + std::to_string(bc.T_in) + ">";
        });

    // SinglePhaseSolver ----------------------------------------------------
    py::class_<SinglePhaseSolver>(m, "SinglePhaseSolver")
        .def(py::init<int, double, double, double, double, double>(),
             py::arg("N"), py::arg("R"), py::arg("C"),
             py::arg("rho"), py::arg("Cp"), py::arg("V"),
             "Construct solver for an N-cell staggered-mesh pipe.\n\n"
             "Parameters\n----------\n"
             "N   : number of cells\n"
             "R   : cell friction resistance [Pa/(kg/s)]\n"
             "C   : cell hydraulic compressibility [kg/Pa]\n"
             "rho : fluid density [kg/m³]\n"
             "Cp  : specific heat [J/(kg·K)]\n"
             "V   : cell volume [m³]")

        .def_property_readonly("N",   &SinglePhaseSolver::N)
        .def_property_readonly("R",   &SinglePhaseSolver::R)
        .def_property_readonly("C",   &SinglePhaseSolver::C)
        .def_property_readonly("rho", &SinglePhaseSolver::rho)
        .def_property_readonly("Cp",  &SinglePhaseSolver::Cp)
        .def_property_readonly("V",   &SinglePhaseSolver::V)

        .def("step",
            [](SinglePhaseSolver& self,
               py::array_t<double> p,
               py::array_t<double> T,
               py::array_t<double> mdot,
               const BoundaryConditions& bc,
               double dt)
            {
                auto p_v    = to_vec(p);
                auto T_v    = to_vec(T);
                auto mdot_v = to_vec(mdot);
                self.step(p_v, T_v, mdot_v, bc, dt);
                copy_back(p,    p_v);
                copy_back(T,    T_v);
                copy_back(mdot, mdot_v);
            },
            py::arg("p"), py::arg("T"), py::arg("mdot"),
            py::arg("bc"), py::arg("dt"),
            "Advance one timestep, modifying p, T, mdot arrays in-place.")

        .def("solve",
            [](const SinglePhaseSolver& self,
               py::array_t<double> p0,
               py::array_t<double> T0,
               py::array_t<double> mdot0,
               const BoundaryConditions& bc,
               double dt, int n_steps, int stride)
            -> py::array_t<double>
            {
                auto p_v    = to_vec(p0);
                auto T_v    = to_vec(T0);
                auto mdot_v = to_vec(mdot0);

                auto flat = self.solve(p_v, T_v, mdot_v, bc, dt, n_steps, stride);

                // Layout per snapshot: p[N] + T[N] + mdot[N+1] = 3N+1 doubles.
                int state_size = 3 * self.N() + 1;
                int n_snap     = static_cast<int>(flat.size()) / state_size;

                // Return shape (n_snap, state_size): [p, T, mdot] per row
                py::array_t<double> result({n_snap, state_size});
                std::memcpy(result.mutable_data(), flat.data(),
                            flat.size() * sizeof(double));
                return result;
            },
            py::arg("p"), py::arg("T"), py::arg("mdot"),
            py::arg("bc"), py::arg("dt"), py::arg("n_steps"),
            py::arg("stride") = 1,
            "Run n_steps timesteps.\n\n"
            "Returns array of shape (n_snapshots, 3*N+1).\n"
            "Each row: [ p[0..N-1], T[0..N-1], mdot[0..N] ]\n"
            "Snapshots taken every `stride` steps.");
}
