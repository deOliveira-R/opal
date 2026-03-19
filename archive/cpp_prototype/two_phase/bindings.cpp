/**
 * bindings.cpp — pybind11 bindings for the two-phase solver.
 *
 * Phase 3 update: exposes FlowModel selection alongside legacy API.
 * All original bindings are preserved for backward compatibility.
 *
 * Exposes:
 *   opal_two_phase.SimpleFluidProperties()
 *   opal_two_phase.IAPWSIF97Properties()
 *   opal_two_phase.PressureFace(p, h_l, h_v, alpha)
 *   opal_two_phase.WallFace(h_l, h_v)
 *   opal_two_phase.BreakFace(p_back, C_d, h_l, h_v)
 *   opal_two_phase.RampedBreak(p_back, C_d_final, t_open, h_l, h_v)
 *   opal_two_phase.MUSCL(limiter="minmod")  # or "van_leer", "superbee", "mc"
 *   opal_two_phase.HEMModel()
 *   opal_two_phase.TwoPhaseSolver(N, dx, A_flow, D_h, f_D, fluid, [recon], [model])
 *     .step(p, h, mdot, bc, dt, q_wall=None)
 *     .solve(p, h, mdot, bc, dt, n_steps, stride=1, q_wall=None)
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "solver.hpp"
#include "simple_fluid.hpp"
#include "iapws97.hpp"
#include "fluid_package.hpp"
#include "reconstruction.hpp"
#include "flow_model.hpp"
#include "hem_model.hpp"
#include "five_eq_model.hpp"
#include "phasic_properties.hpp"
#include "closures.hpp"
#include "momentum.hpp"
#include "critical_flow.hpp"

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
    m.doc() = "OPAL two-phase semi-implicit staggered-mesh solver";

    // Property hierarchy: FluidProperties → FluidPackage (+ PhasicProperties)
    py::class_<FluidProperties>(m, "FluidProperties");
    py::class_<PhasicProperties>(m, "PhasicProperties");
    py::class_<FluidPackage, FluidProperties, PhasicProperties>(m, "FluidPackage")
        .def_property_readonly("p_min", &FluidPackage::p_min)
        .def_property_readonly("p_max", &FluidPackage::p_max);

    // SimpleFluidProperties -------------------------------------------------
    py::class_<SimpleFluidProperties, FluidPackage>(m, "SimpleFluidProperties")
        .def(py::init<>(), "Synthetic linear test fluid matching SimpleFluid.mo")
        .def("evaluate", &SimpleFluidProperties::evaluate,
             py::arg("p"), py::arg("h"),
             "Evaluate all properties at (p, h)")
        .def("evaluate_phasic", &SimpleFluidProperties::evaluate_phasic,
             py::arg("p"),
             "Evaluate phasic (saturation) properties at pressure p")
        .def("rho_liquid", &SimpleFluidProperties::rho_liquid,
             py::arg("p"), py::arg("h_l"))
        .def("rho_vapor", &SimpleFluidProperties::rho_vapor,
             py::arg("p"), py::arg("h_v"))
        .def("T_liquid", &SimpleFluidProperties::T_liquid,
             py::arg("p"), py::arg("h_l"))
        .def("T_vapor", &SimpleFluidProperties::T_vapor,
             py::arg("p"), py::arg("h_v"));

    // IAPWSIF97Properties ---------------------------------------------------
    py::class_<IAPWSIF97Properties, FluidPackage>(m, "IAPWSIF97Properties")
        .def(py::init<>(), "IAPWS-IF97 industrial steam tables (Regions 1, 2, 4)")
        .def("evaluate", &IAPWSIF97Properties::evaluate,
             py::arg("p"), py::arg("h"),
             "Evaluate all properties at (p, h)")
        .def("evaluate_phasic", &IAPWSIF97Properties::evaluate_phasic,
             py::arg("p"))
        .def("rho_liquid", &IAPWSIF97Properties::rho_liquid,
             py::arg("p"), py::arg("h_l"))
        .def("rho_vapor", &IAPWSIF97Properties::rho_vapor,
             py::arg("p"), py::arg("h_v"))
        .def("T_liquid", &IAPWSIF97Properties::T_liquid,
             py::arg("p"), py::arg("h_l"))
        .def("T_vapor", &IAPWSIF97Properties::T_vapor,
             py::arg("p"), py::arg("h_v"));

    // FluidProps struct (returned by evaluate) ------------------------------
    py::class_<FluidProps>(m, "FluidProps")
        .def_readonly("rho",       &FluidProps::rho)
        .def_readonly("drho_dp_h", &FluidProps::drho_dp_h)
        .def_readonly("drho_dh_p", &FluidProps::drho_dh_p)
        .def_readonly("T",         &FluidProps::T);

    // FaceReconstruction hierarchy -----------------------------------------
    py::class_<FaceReconstruction>(m, "FaceReconstruction");

    py::class_<DonorCell, FaceReconstruction>(m, "DonorCell")
        .def(py::init<>(), "First-order upwind (donor cell)");

    // Limiter selection: pass a string name to MUSCL constructor.
    // Supported: "minmod", "van_leer", "superbee", "mc"
    py::class_<MUSCL, FaceReconstruction>(m, "MUSCL")
        .def(py::init([](const std::string& name) {
            if (name == "minmod")        return new MUSCL(limiters::minmod);
            if (name == "van_leer")      return new MUSCL(limiters::van_leer);
            if (name == "superbee")      return new MUSCL(limiters::superbee);
            if (name == "mc")            return new MUSCL(limiters::mc);
            throw std::invalid_argument(
                "Unknown limiter '" + name + "'. "
                "Options: minmod, van_leer, superbee, mc");
        }),
        py::arg("limiter") = "minmod",
        "Second-order TVD MUSCL with selectable slope limiter");

    // FlowModel hierarchy --------------------------------------------------
    py::class_<FlowModel>(m, "FlowModel")
        .def_property_readonly("name", &FlowModel::name)
        .def_property_readonly("vars_per_cell", &FlowModel::vars_per_cell);

    py::class_<HEMModel, FlowModel>(m, "HEMModel")
        .def(py::init<>(), "3-equation Homogeneous Equilibrium Model");

    // PhasicProps struct ---------------------------------------------------
    py::class_<PhasicProps>(m, "PhasicProps")
        .def_readonly("rho_l",       &PhasicProps::rho_l)
        .def_readonly("rho_v",       &PhasicProps::rho_v)
        .def_readonly("h_sat_l",     &PhasicProps::h_sat_l)
        .def_readonly("h_sat_v",     &PhasicProps::h_sat_v)
        .def_readonly("T_sat",       &PhasicProps::T_sat)
        .def_readonly("drho_l_dp",   &PhasicProps::drho_l_dp)
        .def_readonly("drho_v_dp",   &PhasicProps::drho_v_dp)
        .def_readonly("cp_l",        &PhasicProps::cp_l)
        .def_readonly("cp_v",        &PhasicProps::cp_v)
        .def_readonly("sigma",       &PhasicProps::sigma);

    // FaceTransportBC struct (for update_transport) -------------------------
    py::class_<FaceTransportBC>(m, "FaceTransportBC")
        .def(py::init<>())
        .def_readwrite("h_l",    &FaceTransportBC::h_l)
        .def_readwrite("h_v",    &FaceTransportBC::h_v)
        .def_readwrite("h_mix",  &FaceTransportBC::h_mix)
        .def_readwrite("alpha",  &FaceTransportBC::alpha);

    // InterfacialState struct (input to closures) --------------------------
    py::class_<InterfacialState>(m, "InterfacialState")
        .def(py::init<>())
        .def_readwrite("p",       &InterfacialState::p)
        .def_readwrite("alpha",   &InterfacialState::alpha)
        .def_readwrite("rho_l",   &InterfacialState::rho_l)
        .def_readwrite("rho_v",   &InterfacialState::rho_v)
        .def_readwrite("h_l",     &InterfacialState::h_l)
        .def_readwrite("h_v",     &InterfacialState::h_v)
        .def_readwrite("T_l",     &InterfacialState::T_l)
        .def_readwrite("T_v",     &InterfacialState::T_v)
        .def_readwrite("T_sat",   &InterfacialState::T_sat)
        .def_readwrite("h_sat_l", &InterfacialState::h_sat_l)
        .def_readwrite("h_sat_v", &InterfacialState::h_sat_v)
        .def_readwrite("cp_l",    &InterfacialState::cp_l)
        .def_readwrite("sigma",   &InterfacialState::sigma)
        .def_readwrite("D_h",     &InterfacialState::D_h)
        .def_readwrite("g_mag",   &InterfacialState::g_mag);

    // ClosureResult struct (output from closures) -------------------------
    py::class_<ClosureResult>(m, "ClosureResult")
        .def_readonly("Gamma", &ClosureResult::Gamma)
        .def_readonly("q_i_l", &ClosureResult::q_i_l)
        .def_readonly("q_i_v", &ClosureResult::q_i_v);

    // DriftFluxResult struct (output from drift_flux) --------------------
    py::class_<DriftFluxResult>(m, "DriftFluxResult")
        .def_readonly("C_0",  &DriftFluxResult::C_0)
        .def_readonly("V_gj", &DriftFluxResult::V_gj);

    // Sub-model interfaces --------------------------------------------------
    py::class_<HeatTransferModel>(m, "HeatTransferModel")
        .def("evaluate", &HeatTransferModel::evaluate,
             py::arg("state"), "Compute interfacial transfer rates");

    py::class_<DriftVelocityModel>(m, "DriftVelocityModel")
        .def("evaluate", &DriftVelocityModel::evaluate,
             py::arg("state"), "Compute drift-flux parameters");

    // Concrete sub-models ---------------------------------------------------
    py::class_<LinearRelaxation, HeatTransferModel>(m, "LinearRelaxation")
        .def(py::init<double, double>(),
             py::arg("H_i") = 1e5, py::arg("alpha_nucleation") = 1e-3,
             "Linear relaxation: q_i = H_i * a_i * (T_sat - T_l)")
        .def_property_readonly("H_i", &LinearRelaxation::H_i)
        .def_property_readonly("alpha_nucleation", &LinearRelaxation::alpha_nucleation);

    py::class_<ZuberFindlay, DriftVelocityModel>(m, "ZuberFindlay")
        .def(py::init<double>(),
             py::arg("C_0") = 1.13,
             "Zuber-Findlay drift velocity for churn-turbulent bubbly flow")
        .def_property_readonly("C_0", &ZuberFindlay::C_0);

    // InterfacialClosures hierarchy ----------------------------------------
    py::class_<InterfacialClosures>(m, "InterfacialClosures")
        .def("compute", &InterfacialClosures::compute,
             py::arg("state"), "Compute interfacial transfer rates")
        .def("drift_flux", &InterfacialClosures::drift_flux,
             py::arg("state"), "Compute drift-flux parameters");

    py::class_<NoClosures, InterfacialClosures>(m, "NoClosures")
        .def(py::init<>(), "No closures (for HEM)");

    py::class_<DriftFluxClosures, InterfacialClosures>(m, "DriftFluxClosures")
        .def(py::init<const HeatTransferModel&, const DriftVelocityModel&>(),
             py::arg("heat_transfer"), py::arg("drift_velocity"),
             py::keep_alive<1, 2>(),  // closures keeps ht alive
             py::keep_alive<1, 3>(),  // closures keeps drift alive
             "Drift-flux closures with pluggable sub-models");

    // FiveEqModel ----------------------------------------------------------
    py::class_<FiveEqModel, FlowModel>(m, "FiveEqModel")
        .def(py::init<const PhasicProperties&, const InterfacialClosures&>(),
             py::arg("phasic"), py::arg("closures"),
             py::keep_alive<1, 2>(),  // model keeps phasic alive
             py::keep_alive<1, 3>(),  // model keeps closures alive
             "5-equation drift-flux model")

        // Transport-only update: given already-solved p and mdot from a
        // Python-level pressure/momentum solve, update alpha/h_l/h_v using
        // the C++ closures (nucleation, interfacial HT, enthalpy bounds).
        .def("update_transport",
            [](const FiveEqModel& self,
               py::array_t<double> p,
               py::array_t<double> p_old_arr,
               py::array_t<double> alpha,
               py::array_t<double> h_l,
               py::array_t<double> h_v,
               py::array_t<double> mdot,
               const FaceTransportBC& tbc_in,
               int N, double dx, double A_flow, double D_h, double f_D,
               double dt,
               py::object q_wall_obj)
            {
                SolverState state;
                state.p     = to_vec(p);
                state.alpha = to_vec(alpha);
                state.h_l   = to_vec(h_l);
                state.h_v   = to_vec(h_v);
                state.mdot  = to_vec(mdot);

                auto p_old_vec = to_vec(p_old_arr);
                MeshParams mesh{N, dx, A_flow, D_h, f_D, dx * A_flow, 9.81, 9.81};
                std::vector<FluidProps> props(N);
                static const DonorCell default_recon;
                static const SolverNumerics default_numerics;

                if (q_wall_obj.is_none()) {
                    self.update_transport(
                        state, p_old_vec, tbc_in, mesh, props, default_recon, dt, nullptr, default_numerics);
                } else {
                    auto q_v = to_vec(q_wall_obj.cast<py::array_t<double>>());
                    self.update_transport(
                        state, p_old_vec, tbc_in, mesh, props, default_recon, dt, &q_v, default_numerics);
                }

                copy_back(alpha, state.alpha);
                copy_back(h_l,   state.h_l);
                copy_back(h_v,   state.h_v);
            },
            py::arg("p"), py::arg("p_old"),
            py::arg("alpha"), py::arg("h_l"), py::arg("h_v"),
            py::arg("mdot"), py::arg("tbc_in"),
            py::arg("N"), py::arg("dx"), py::arg("A_flow"),
            py::arg("D_h"), py::arg("f_D"),
            py::arg("dt"),
            py::arg("q_wall") = py::none(),
            "Transport-only: update alpha/h_l/h_v using C++ closures.")

        // Phasic flux split (public for unit testing)
        .def("split_phasic_flux", &FiveEqModel::split_phasic_flux,
             py::arg("mdot_m"), py::arg("alpha_face"),
             py::arg("rho_l"), py::arg("rho_v"), py::arg("rho_m"),
             py::arg("C_0"), py::arg("V_gj"), py::arg("A_flow"),
             "Split mixture mass flux into (mdot_l, mdot_v) via drift-flux.")

        // Direct 5-eq step: operates on (p, alpha, h_l, h_v, mdot) arrays
        .def("make_state_5eq",
            [](const FiveEqModel& self,
               py::array_t<double> p,
               py::array_t<double> alpha,
               py::array_t<double> h_l,
               py::array_t<double> h_v,
               py::array_t<double> mdot) -> py::dict
            {
                auto s = self.make_state_5eq(
                    to_vec(p), to_vec(alpha), to_vec(h_l),
                    to_vec(h_v), to_vec(mdot));
                py::dict d;
                d["p"] = py::array_t<double>(s.p.size(), s.p.data());
                d["alpha"] = py::array_t<double>(s.alpha.size(), s.alpha.data());
                d["h_l"] = py::array_t<double>(s.h_l.size(), s.h_l.data());
                d["h_v"] = py::array_t<double>(s.h_v.size(), s.h_v.data());
                d["mdot"] = py::array_t<double>(s.mdot.size(), s.mdot.data());
                return d;
            });

    // BCType REMOVED — replaced by FacePressureBC::Type (DIRICHLET/ZERO_FLUX)

    // SourceTerms -----------------------------------------------------------
    py::class_<SourceTerms>(m, "SourceTerms")
        .def(py::init<>())
        .def_readwrite("mass",      &SourceTerms::mass)
        .def_readwrite("energy_l",  &SourceTerms::energy_l)
        .def_readwrite("energy_v",  &SourceTerms::energy_v)
        .def_readwrite("void_frac", &SourceTerms::void_frac)
        .def_readwrite("momentum",  &SourceTerms::momentum)
        .def("empty", &SourceTerms::empty);

    // BoundaryConditions REMOVED — replaced by BoundaryFace hierarchy

    // BoundaryFace hierarchy -----------------------------------------------
    py::class_<BoundaryFace>(m, "BoundaryFace");

    py::class_<PressureFace, BoundaryFace>(m, "PressureFace")
        .def(py::init<double, double, double, double>(),
             py::arg("p"), py::arg("h_l"),
             py::arg("h_v") = 0.0, py::arg("alpha") = 0.0,
             "Pressure BC: specified pressure + inflow enthalpy");

    py::class_<WallFace, BoundaryFace>(m, "WallFace")
        .def(py::init<double, double>(),
             py::arg("h_l") = 0.0, py::arg("h_v") = 0.0,
             "Wall BC: zero flux, no pressure coupling");

    py::class_<BreakFace, BoundaryFace>(m, "BreakFace")
        .def(py::init<double, double, double, double>(),
             py::arg("p_back"), py::arg("C_d"),
             py::arg("h_l") = 0.0, py::arg("h_v") = 0.0,
             "Break BC: pressure BC with critical flow limiter");

    py::class_<RampedBreak, BreakFace>(m, "RampedBreak")
        .def(py::init<double, double, double, double, double>(),
             py::arg("p_back"), py::arg("C_d_final"), py::arg("t_open"),
             py::arg("h_l") = 0.0, py::arg("h_v") = 0.0,
             "Time-ramped break: C_d ramps from 0 to C_d_final over t_open seconds");

    // SolverNumerics --------------------------------------------------------
    py::class_<SolverNumerics>(m, "SolverNumerics")
        .def(py::init<>())
        .def_readwrite("alpha_min",              &SolverNumerics::alpha_min)
        .def_readwrite("rv_floor_frac",          &SolverNumerics::rv_floor_frac)
        .def_readwrite("rv_floor_abs",           &SolverNumerics::rv_floor_abs)
        .def_readwrite("alpha_nucleation_floor", &SolverNumerics::alpha_nucleation_floor)
        .def_readwrite("m_phase_min",            &SolverNumerics::m_phase_min)
        .def_readwrite("h_l_min",                &SolverNumerics::h_l_min)
        .def_readwrite("h_v_max",                &SolverNumerics::h_v_max)
        .def_readwrite("rho_face_min",           &SolverNumerics::rho_face_min);

    // FrictionModel hierarchy -----------------------------------------------
    py::class_<FrictionModel>(m, "FrictionModel");

    py::class_<DarcyFriction, FrictionModel>(m, "DarcyFriction")
        .def(py::init<double>(),
             py::arg("rho_face_min") = 0.01,
             "Single-phase Darcy-Weisbach friction")
        .def_property_readonly("rho_face_min", &DarcyFriction::rho_face_min);

    // MomentumModel hierarchy ----------------------------------------------
    py::class_<MomentumModel>(m, "MomentumModel")
        .def_property_readonly("name", &MomentumModel::name);

    py::class_<AlgebraicMomentum, MomentumModel>(m, "AlgebraicMomentum")
        .def(py::init<>(), "Steady-state algebraic momentum: mdot = dp/R");

    py::class_<InertialMomentum, MomentumModel>(m, "InertialMomentum")
        .def(py::init<>(), "Time-advanced inertial momentum (default Darcy friction)")
        .def(py::init<const FrictionModel&>(),
             py::arg("friction"),
             py::keep_alive<1, 2>(),
             "Time-advanced inertial momentum with custom friction model");

    // CriticalFlowModel hierarchy ------------------------------------------
    // CriticalFlowResult struct
    py::class_<CriticalFlowResult>(m, "CriticalFlowResult")
        .def_readonly("mdot_crit", &CriticalFlowResult::mdot_crit)
        .def_readonly("is_choked", &CriticalFlowResult::is_choked);

    py::class_<CriticalFlowModel>(m, "CriticalFlowModel")
        .def_property_readonly("name", &CriticalFlowModel::name)
        .def("evaluate", &CriticalFlowModel::evaluate,
             py::arg("p_cell"), py::arg("h_mix"), py::arg("rho"),
             py::arg("drho_dp_h"), py::arg("p_back"), py::arg("A_flow"),
             py::arg("C_d"), py::arg("mdot_momentum"));

    py::class_<NoCriticalFlow, CriticalFlowModel>(m, "NoCriticalFlow")
        .def(py::init<>(), "No critical flow (never choked)");

    py::class_<RansomTrapp, CriticalFlowModel>(m, "RansomTrapp")
        .def(py::init<const FluidPackage&, double, double>(),
             py::arg("phasic"),
             py::arg("x_trans") = 0.10, py::arg("c_floor") = 1200.0,
             py::keep_alive<1, 2>(),  // keeps phasic alive
             "Ransom-Trapp critical flow with internal saturation lookup");

    // TwoPhaseBCs REMOVED — replaced by BoundaryFace hierarchy

    // TwoPhaseSolver --------------------------------------------------------
    py::class_<TwoPhaseSolver>(m, "TwoPhaseSolver")
        // Legacy constructor (no recon, no model — uses DonorCell + HEM)
        .def(py::init<int, double, double, double, double,
                      const FluidPackage&>(),
             py::arg("N"), py::arg("dx"), py::arg("A_flow"),
             py::arg("D_h"), py::arg("f_D"), py::arg("fluid"),
             py::keep_alive<1, 7>(),
             "Construct solver with first-order donor-cell and HEM model.")

        // Legacy constructor with reconstruction (no model — uses HEM)
        .def(py::init<int, double, double, double, double,
                      const FluidPackage&, const FaceReconstruction&>(),
             py::arg("N"), py::arg("dx"), py::arg("A_flow"),
             py::arg("D_h"), py::arg("f_D"), py::arg("fluid"),
             py::arg("recon"),
             py::keep_alive<1, 7>(),  // solver keeps fluid alive
             py::keep_alive<1, 8>(),  // solver keeps recon alive
             "Construct solver with selectable reconstruction and HEM model.")

        // New constructor with model selection
        .def(py::init<int, double, double, double, double,
                      const FluidPackage&, const FaceReconstruction&,
                      const FlowModel&>(),
             py::arg("N"), py::arg("dx"), py::arg("A_flow"),
             py::arg("D_h"), py::arg("f_D"), py::arg("fluid"),
             py::arg("recon"), py::arg("model"),
             py::keep_alive<1, 7>(),  // solver keeps fluid alive
             py::keep_alive<1, 8>(),  // solver keeps recon alive
             py::keep_alive<1, 9>(),  // solver keeps model alive
             "Construct solver with selectable reconstruction and flow model.")

        // Full constructor with momentum + critical flow
        .def(py::init<int, double, double, double, double,
                      const FluidPackage&, const FaceReconstruction&,
                      const FlowModel&, const MomentumModel&,
                      const CriticalFlowModel*>(),
             py::arg("N"), py::arg("dx"), py::arg("A_flow"),
             py::arg("D_h"), py::arg("f_D"), py::arg("fluid"),
             py::arg("recon"), py::arg("model"),
             py::arg("momentum"),
             py::arg("critical_flow") = nullptr,
             py::keep_alive<1, 7>(),   // fluid
             py::keep_alive<1, 8>(),   // recon
             py::keep_alive<1, 9>(),   // model
             py::keep_alive<1, 10>(),  // momentum
             py::keep_alive<1, 11>(),  // critical_flow
             "Full constructor: flow model + momentum + critical flow.")

        .def_property_readonly("N",      &TwoPhaseSolver::N)
        .def_property_readonly("dx",     &TwoPhaseSolver::dx)
        .def_property_readonly("A_flow", &TwoPhaseSolver::A_flow)
        .def_property_readonly("D_h",    &TwoPhaseSolver::D_h)
        .def_property_readonly("f_D",    &TwoPhaseSolver::f_D)
        .def_property_readonly("V",      &TwoPhaseSolver::V)
        .def("set_gravity", &TwoPhaseSolver::set_gravity,
             py::arg("g_axial"), py::arg("g_mag"),
             "Set gravity: g_axial = projection on pipe axis, g_mag = |g| for buoyancy")

        // step_5eq REMOVED — all callers migrated to step_bf

        // BoundaryFace-based step: time-aware, strategy BCs
        .def("step_bf",
            [](TwoPhaseSolver& self,
               py::array_t<double> p,
               py::array_t<double> alpha,
               py::array_t<double> h_l,
               py::array_t<double> h_v,
               py::array_t<double> mdot,
               const BoundaryFace& bc_in,
               const BoundaryFace& bc_out,
               double t, double dt,
               py::object q_wall_obj,
               py::object sources_obj)
            {
                SolverState state;
                state.p     = to_vec(p);
                state.alpha = to_vec(alpha);
                state.h_l   = to_vec(h_l);
                state.h_v   = to_vec(h_v);
                state.mdot  = to_vec(mdot);

                const SourceTerms* src_ptr = nullptr;
                SourceTerms src;
                if (!sources_obj.is_none()) {
                    src = sources_obj.cast<SourceTerms>();
                    src_ptr = &src;
                }

                if (q_wall_obj.is_none()) {
                    self.step(state, bc_in, bc_out, t, dt, nullptr, src_ptr);
                } else {
                    auto q_v = to_vec(q_wall_obj.cast<py::array_t<double>>());
                    self.step(state, bc_in, bc_out, t, dt, &q_v, src_ptr);
                }

                copy_back(p,     state.p);
                copy_back(alpha, state.alpha);
                copy_back(h_l,   state.h_l);
                copy_back(h_v,   state.h_v);
                copy_back(mdot,  state.mdot);
            },
            py::arg("p"), py::arg("alpha"), py::arg("h_l"), py::arg("h_v"),
            py::arg("mdot"),
            py::arg("bc_in"), py::arg("bc_out"),
            py::arg("t"), py::arg("dt"),
            py::arg("q_wall") = py::none(),
            py::arg("sources") = py::none(),
            "Step with BoundaryFace strategy objects (time-aware BCs).")

        // HEM step via BoundaryFace: 3-variable (p, h, mdot) interface
        .def("step_hem_bf",
            [](TwoPhaseSolver& self,
               py::array_t<double> p,
               py::array_t<double> h,
               py::array_t<double> mdot,
               const BoundaryFace& bc_in,
               const BoundaryFace& bc_out,
               double t, double dt,
               py::object q_wall_obj)
            {
                int N = self.N();
                auto p_v = to_vec(p); auto h_v = to_vec(h); auto m_v = to_vec(mdot);
                if (static_cast<int>(p_v.size()) != N)
                    throw std::invalid_argument("p size mismatch");
                if (static_cast<int>(h_v.size()) != N)
                    throw std::invalid_argument("h size mismatch");
                if (static_cast<int>(m_v.size()) != N + 1)
                    throw std::invalid_argument("mdot size mismatch");
                SolverState state = self.model().make_state(p_v, h_v, m_v);

                if (q_wall_obj.is_none()) {
                    self.step(state, bc_in, bc_out, t, dt);
                } else {
                    auto q_v = to_vec(q_wall_obj.cast<py::array_t<double>>());
                    self.step(state, bc_in, bc_out, t, dt, &q_v);
                }

                copy_back(p,    state.p);
                copy_back(h,    state.h_l);
                copy_back(mdot, state.mdot);
            },
            py::arg("p"), py::arg("h"), py::arg("mdot"),
            py::arg("bc_in"), py::arg("bc_out"),
            py::arg("t"), py::arg("dt"),
            py::arg("q_wall") = py::none(),
            "HEM step with BoundaryFace BCs (3-variable: p, h, mdot).")

        // Legacy solve
        // solve() REMOVED — all callers migrated to step_bf/step_hem_bf loops
        ;
}
