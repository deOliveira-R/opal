/**
 * bindings.cpp — pybind11 bindings for the two-phase solver.
 *
 * Exposes:
 *   opal_two_phase.SimpleFluidProperties()
 *   opal_two_phase.TwoPhaseBCs(p_in, p_out, h_in)
 *   opal_two_phase.TwoPhaseSolver(N, dx, A_flow, D_h, f_D, fluid)
 *     .step(p, h, mdot, bc, dt, q_wall=None)
 *     .solve(p, h, mdot, bc, dt, n_steps, stride=1, q_wall=None)
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "solver.hpp"
#include "simple_fluid.hpp"
#include "iapws97.hpp"

namespace py = pybind11;
using namespace opal;

// ---------------------------------------------------------------------------
// Helpers — numpy <-> std::vector conversion
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

PYBIND11_MODULE(opal_two_phase, m) {
    m.doc() = "OPAL Phase 2 two-phase semi-implicit staggered-mesh solver";

    // FluidProperties base (abstract, not directly constructible) -----------
    py::class_<FluidProperties>(m, "FluidProperties");

    // SimpleFluidProperties -------------------------------------------------
    py::class_<SimpleFluidProperties, FluidProperties>(m, "SimpleFluidProperties")
        .def(py::init<>(), "Synthetic linear test fluid matching SimpleFluid.mo")
        .def("evaluate", &SimpleFluidProperties::evaluate,
             py::arg("p"), py::arg("h"),
             "Evaluate all properties at (p, h)");

    // IAPWSIF97Properties ---------------------------------------------------
    py::class_<IAPWSIF97Properties, FluidProperties>(m, "IAPWSIF97Properties")
        .def(py::init<>(), "IAPWS-IF97 industrial steam tables (Regions 1, 2, 4)")
        .def("evaluate", &IAPWSIF97Properties::evaluate,
             py::arg("p"), py::arg("h"),
             "Evaluate all properties at (p, h)");

    // FluidProps struct (returned by evaluate) ------------------------------
    py::class_<FluidProps>(m, "FluidProps")
        .def_readonly("rho",       &FluidProps::rho)
        .def_readonly("drho_dp_h", &FluidProps::drho_dp_h)
        .def_readonly("drho_dh_p", &FluidProps::drho_dh_p)
        .def_readonly("T",         &FluidProps::T);

    // TwoPhaseBCs -----------------------------------------------------------
    py::class_<TwoPhaseBCs>(m, "TwoPhaseBCs")
        .def(py::init<double, double, double>(),
             py::arg("p_in"), py::arg("p_out"), py::arg("h_in"),
             "Boundary conditions: inlet/outlet pressure [Pa], inlet enthalpy [J/kg]")
        .def_readwrite("p_in",  &TwoPhaseBCs::p_in)
        .def_readwrite("p_out", &TwoPhaseBCs::p_out)
        .def_readwrite("h_in",  &TwoPhaseBCs::h_in)
        .def("__repr__", [](const TwoPhaseBCs& bc) {
            return "<TwoPhaseBCs p_in=" + std::to_string(bc.p_in) +
                   " p_out=" + std::to_string(bc.p_out) +
                   " h_in=" + std::to_string(bc.h_in) + ">";
        });

    // TwoPhaseSolver --------------------------------------------------------
    py::class_<TwoPhaseSolver>(m, "TwoPhaseSolver")
        .def(py::init<int, double, double, double, double,
                      const FluidProperties&>(),
             py::arg("N"), py::arg("dx"), py::arg("A_flow"),
             py::arg("D_h"), py::arg("f_D"), py::arg("fluid"),
             py::keep_alive<1, 7>(),  // solver (1) keeps fluid (7) alive
             "Construct solver for an N-cell two-phase staggered-mesh pipe.\n\n"
             "Parameters\n----------\n"
             "N      : number of cells\n"
             "dx     : cell length [m]\n"
             "A_flow : flow area [m^2]\n"
             "D_h    : hydraulic diameter [m]\n"
             "f_D    : Darcy friction factor [-]\n"
             "fluid  : FluidProperties instance")

        .def_property_readonly("N",      &TwoPhaseSolver::N)
        .def_property_readonly("dx",     &TwoPhaseSolver::dx)
        .def_property_readonly("A_flow", &TwoPhaseSolver::A_flow)
        .def_property_readonly("D_h",    &TwoPhaseSolver::D_h)
        .def_property_readonly("f_D",    &TwoPhaseSolver::f_D)
        .def_property_readonly("V",      &TwoPhaseSolver::V)

        .def("step",
            [](TwoPhaseSolver& self,
               py::array_t<double> p,
               py::array_t<double> h,
               py::array_t<double> mdot,
               const TwoPhaseBCs& bc,
               double dt,
               py::object q_wall_obj)
            {
                auto p_v    = to_vec(p);
                auto h_v    = to_vec(h);
                auto mdot_v = to_vec(mdot);

                if (q_wall_obj.is_none()) {
                    self.step(p_v, h_v, mdot_v, bc, dt);
                } else {
                    auto q_v = to_vec(q_wall_obj.cast<py::array_t<double>>());
                    self.step(p_v, h_v, mdot_v, bc, dt, &q_v);
                }

                copy_back(p,    p_v);
                copy_back(h,    h_v);
                copy_back(mdot, mdot_v);
            },
            py::arg("p"), py::arg("h"), py::arg("mdot"),
            py::arg("bc"), py::arg("dt"),
            py::arg("q_wall") = py::none(),
            "Advance one timestep, modifying p, h, mdot arrays in-place.")

        .def("solve",
            [](const TwoPhaseSolver& self,
               py::array_t<double> p0,
               py::array_t<double> h0,
               py::array_t<double> mdot0,
               const TwoPhaseBCs& bc,
               double dt, int n_steps, int stride,
               py::object q_wall_obj)
            -> py::array_t<double>
            {
                auto p_v    = to_vec(p0);
                auto h_v    = to_vec(h0);
                auto mdot_v = to_vec(mdot0);

                std::vector<double> flat;
                if (q_wall_obj.is_none()) {
                    flat = self.solve(p_v, h_v, mdot_v, bc, dt, n_steps, stride);
                } else {
                    auto q_v = to_vec(q_wall_obj.cast<py::array_t<double>>());
                    flat = self.solve(p_v, h_v, mdot_v, bc, dt, n_steps, stride, &q_v);
                }

                int state_size = 3 * self.N() + 1;
                int n_snap     = static_cast<int>(flat.size()) / state_size;

                py::array_t<double> result({n_snap, state_size});
                std::memcpy(result.mutable_data(), flat.data(),
                            flat.size() * sizeof(double));
                return result;
            },
            py::arg("p"), py::arg("h"), py::arg("mdot"),
            py::arg("bc"), py::arg("dt"), py::arg("n_steps"),
            py::arg("stride") = 1,
            py::arg("q_wall") = py::none(),
            "Run n_steps timesteps.\n\n"
            "Returns array of shape (n_snapshots, 3*N+1).\n"
            "Each row: [ p[0..N-1], h[0..N-1], mdot[0..N] ]\n"
            "Snapshots taken every `stride` steps.");
}
