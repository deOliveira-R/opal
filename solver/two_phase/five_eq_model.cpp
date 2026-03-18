/**
 * five_eq_model.cpp — 5-equation drift-flux FlowModel implementation.
 *
 * Derived in: derivations/five_eq_pressure_linearization.py
 *             derivations/five_eq_phasic_energy.py
 */

#include "five_eq_model.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace opal {

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

FiveEqModel::FiveEqModel(const PhasicProperties& phasic,
                         const InterfacialClosures& closures)
    : phasic_(phasic), closures_(closures)
{
}

// ---------------------------------------------------------------------------
// State management
// ---------------------------------------------------------------------------

SolverState FiveEqModel::make_state(
    const std::vector<double>& p,
    const std::vector<double>& h,
    const std::vector<double>& mdot) const
{
    // Backward compat: interpret h as mixture, derive alpha and phasic h
    int N = static_cast<int>(p.size());
    SolverState s;
    s.p.resize(N);
    s.alpha.resize(N);
    s.h_l.resize(N);
    s.h_v.resize(N);
    s.mdot = mdot;

    for (int i = 0; i < N; ++i) {
        s.p[i] = p[i];
        auto pp = phasic_.evaluate_phasic(p[i]);

        if (h[i] <= pp.h_sat_l) {
            // Subcooled: all liquid
            s.alpha[i] = 0.0;
            s.h_l[i] = h[i];
            s.h_v[i] = pp.h_sat_v;
        } else if (h[i] >= pp.h_sat_v) {
            // Superheated: all vapor
            s.alpha[i] = 1.0;
            s.h_l[i] = pp.h_sat_l;
            s.h_v[i] = h[i];
        } else {
            // Two-phase: h_l = h_sat_l, h_v = h_sat_v, alpha from quality
            double x = (h[i] - pp.h_sat_l) / (pp.h_sat_v - pp.h_sat_l);
            double v_f = 1.0 / pp.rho_l;
            double v_g = 1.0 / pp.rho_v;
            double v_m = x * v_g + (1.0 - x) * v_f;
            s.alpha[i] = x * v_g / v_m;
            s.h_l[i] = pp.h_sat_l;
            s.h_v[i] = pp.h_sat_v;
        }
    }
    return s;
}

SolverState FiveEqModel::make_state_5eq(
    const std::vector<double>& p,
    const std::vector<double>& alpha,
    const std::vector<double>& h_l,
    const std::vector<double>& h_v,
    const std::vector<double>& mdot) const
{
    SolverState s;
    s.p = p;
    s.alpha = alpha;
    s.h_l = h_l;
    s.h_v = h_v;
    s.mdot = mdot;
    return s;
}

void FiveEqModel::pack_state(
    const SolverState& state,
    std::vector<double>& out) const
{
    int N = static_cast<int>(state.p.size());
    out.clear();
    out.reserve(4 * N + N + 1);
    out.insert(out.end(), state.p.begin(), state.p.end());
    out.insert(out.end(), state.alpha.begin(), state.alpha.end());
    out.insert(out.end(), state.h_l.begin(), state.h_l.end());
    out.insert(out.end(), state.h_v.begin(), state.h_v.end());
    out.insert(out.end(), state.mdot.begin(), state.mdot.end());
}

SolverState FiveEqModel::unpack_state(const double* data, int N) const
{
    SolverState s;
    s.p.assign(data, data + N);
    s.alpha.assign(data + N, data + 2 * N);
    s.h_l.assign(data + 2 * N, data + 3 * N);
    s.h_v.assign(data + 3 * N, data + 4 * N);
    s.mdot.assign(data + 4 * N, data + 5 * N + 1);
    return s;
}

// ---------------------------------------------------------------------------
// Phasic state computation
// ---------------------------------------------------------------------------

void FiveEqModel::compute_phasic_state(
    const SolverState& state,
    const MeshParams& mesh) const
{
    int N = mesh.N;
    phasic_props_.resize(N);
    iface_states_.resize(N);
    closure_results_.resize(N);
    drift_results_.resize(N);

    for (int i = 0; i < N; ++i) {
        phasic_props_[i] = phasic_.evaluate_phasic(state.p[i]);
        auto& pp = phasic_props_[i];

        InterfacialState& is = iface_states_[i];
        is.p      = state.p[i];
        is.alpha  = state.alpha[i];
        is.rho_l  = phasic_.rho_liquid(state.p[i], state.h_l[i]);
        is.rho_v  = phasic_.rho_vapor(state.p[i], state.h_v[i]);
        is.h_l    = state.h_l[i];
        is.h_v    = state.h_v[i];
        is.T_l    = phasic_.T_liquid(state.p[i], state.h_l[i]);
        is.T_v    = phasic_.T_vapor(state.p[i], state.h_v[i]);
        is.T_sat  = pp.T_sat;
        is.h_sat_l = pp.h_sat_l;
        is.h_sat_v = pp.h_sat_v;
        is.cp_l   = pp.cp_l;
        is.sigma  = pp.sigma;
        is.D_h    = mesh.D_h;

        closure_results_[i] = closures_.compute(is);
        drift_results_[i]   = closures_.drift_flux(is);
    }
}

// ---------------------------------------------------------------------------
// Property evaluation (for pressure matrix — mixture properties)
// ---------------------------------------------------------------------------

void FiveEqModel::evaluate_properties(
    const SolverState& state,
    const FluidPackage& fluid,
    std::vector<FluidProps>& props) const
{
    int N = static_cast<int>(state.p.size());
    props.resize(N);

    for (int i = 0; i < N; ++i) {
        // Compute mixture enthalpy for property evaluation
        double al = state.alpha[i];
        double h_mix = (1.0 - al) * state.h_l[i]
                     + al * (state.h_v.empty() ? state.h_l[i] : state.h_v[i]);

        // Use the FluidProperties evaluate at (p, h_mix) to get the correct
        // mixture density and compressibility. This gives the actual drho_dp_h
        // at the subcooled/superheated/two-phase state, NOT the saturation-
        // curve derivative. The pressure equation needs this for stability
        // with inertial momentum during rapid transients.
        props[i] = fluid.evaluate(state.p[i], h_mix);
    }
}

// ---------------------------------------------------------------------------
// Face resistance (same as HEM — mixture density based)
// ---------------------------------------------------------------------------

void FiveEqModel::compute_face_resistance(
    const SolverState& /*state*/,
    const BoundaryConditions& bc,
    const FluidPackage& /*fluid*/,
    const MeshParams& mesh,
    const std::vector<FluidProps>& props,
    std::vector<double>& R_face) const
{
    int N = mesh.N;
    R_face.resize(N + 1);

    double geom = mesh.f_D * mesh.dx / (2.0 * mesh.D_h * mesh.A_flow * mesh.A_flow);

    // Inlet face: compute mixture density from phasic properties at BC
    double rho_l_in = phasic_.rho_liquid(bc.p_in, bc.h_l_in > 0 ? bc.h_l_in : bc.h_in);
    double rho_v_in = phasic_.rho_vapor(bc.p_in, bc.h_v_in > 0 ? bc.h_v_in : bc.h_in);
    double rho_in = (1.0 - bc.alpha_in) * rho_l_in + bc.alpha_in * rho_v_in;
    R_face[0] = geom / (0.5 * (rho_in + props[0].rho));

    for (int i = 1; i < N; ++i) {
        R_face[i] = geom / (0.5 * (props[i - 1].rho + props[i].rho));
    }

    R_face[N] = geom / props[N - 1].rho;
}

// ---------------------------------------------------------------------------
// Pressure system (same structure as HEM — tridiagonal)
// Derived in: derivations/five_eq_pressure_linearization.py
// ---------------------------------------------------------------------------

void FiveEqModel::assemble_pressure_system(
    const SolverState& state,
    const BoundaryConditions& bc,
    const MeshParams& mesh,
    const std::vector<FluidProps>& props,
    const std::vector<double>& R_face,
    double dt,
    TridiagCoeffs& tri) const
{
    int N = mesh.N;
    tri.resize(N);

    for (int i = 0; i < N; ++i) {
        // Derived: a_ii = V/dt * [(1-α)·∂ρ_l/∂P + α·∂ρ_v/∂P]
        double alpha_coeff = mesh.V * props[i].drho_dp_h / dt;
        double inv_R_left  = 1.0 / R_face[i];
        double inv_R_right = 1.0 / R_face[i + 1];

        tri.a[i] = (i > 0)     ? -inv_R_left  : 0.0;
        tri.c[i] = (i < N - 1) ? -inv_R_right : 0.0;
        tri.b[i] = alpha_coeff + inv_R_left + inv_R_right;
        tri.d[i] = alpha_coeff * state.p[i];

        if (i == 0)     tri.d[i] += bc.p_in  * inv_R_left;
        if (i == N - 1) tri.d[i] += bc.p_out * inv_R_right;
    }
}

// ---------------------------------------------------------------------------
// Algebraic flow update (same as HEM)
// ---------------------------------------------------------------------------

void FiveEqModel::update_velocities(
    SolverState& state,
    const BoundaryConditions& bc,
    const MeshParams& mesh,
    const std::vector<double>& R_face) const
{
    int N = mesh.N;
    state.mdot[0] = (bc.p_in - state.p[0]) / R_face[0];
    for (int i = 1; i < N; ++i) {
        state.mdot[i] = (state.p[i - 1] - state.p[i]) / R_face[i];
    }
    state.mdot[N] = (state.p[N - 1] - bc.p_out) / R_face[N];
}

// ---------------------------------------------------------------------------
// Drift-flux phasic flux split
// Derived in: derivations/five_eq_pressure_linearization.py, Step 5
// ---------------------------------------------------------------------------

std::pair<double, double> FiveEqModel::split_phasic_flux(
    double mdot_m, double alpha_face,
    double rho_l, double rho_v, double /*rho_m*/,
    double C_0, double V_gj, double A_flow) const
{
    // Handle extreme alpha: avoid division by near-zero phase fraction
    constexpr double alpha_min = 1e-6;
    if (alpha_face < alpha_min) {
        // All liquid: vapor flow is zero
        return {mdot_m, 0.0};
    }
    if (alpha_face > 1.0 - alpha_min) {
        // All vapor: liquid flow is zero
        return {0.0, mdot_m};
    }

    // G_m = mdot_m / A_flow (mixture mass flux)
    double G_m = mdot_m / A_flow;

    // j = [G_m - α·V_gj·(ρ_v - ρ_l)] / [ρ_l + α·C_0·(ρ_v - ρ_l)]
    double drho = rho_v - rho_l;
    double rho_eff = rho_l + alpha_face * C_0 * drho;
    if (std::abs(rho_eff) < 1.0) {
        rho_eff = std::copysign(1.0, rho_eff);
    }

    double j = (G_m - alpha_face * V_gj * drho) / rho_eff;

    // v_v = C_0·j + V_gj
    double v_v = C_0 * j + V_gj;
    // v_l = [j - α·v_v] / (1-α)
    double alpha_l = 1.0 - alpha_face;
    double v_l = (j - alpha_face * v_v) / alpha_l;

    // Phasic mass flow rates
    double mdot_l = alpha_l * rho_l * v_l * A_flow;
    double mdot_v = alpha_face * rho_v * v_v * A_flow;

    return {mdot_l, mdot_v};
}

// ---------------------------------------------------------------------------
// Explicit transport: void fraction + phasic enthalpies
// Derived in: derivations/five_eq_phasic_energy.py
//
// Enthalpy advection uses the "relative" form:
//   flux = mdot_in*(h_face_in - h_cell) - mdot_out*(h_face_out - h_cell)
// instead of the conservative form mdot*h_face. Both are mathematically
// equivalent when the pressure solve has already enforced mass conservation
// (the h_cell*(mdot_in - mdot_out) part is implicit in the pressure
// equation). The relative form has smaller round-off for near-uniform
// enthalpy, which is the common case in practice.
// ---------------------------------------------------------------------------

void FiveEqModel::update_transport(
    SolverState& state,
    const std::vector<double>& p_old,
    const BoundaryConditions& bc,
    const MeshParams& mesh,
    const std::vector<FluidProps>& /*props*/,
    const FaceReconstruction& recon,
    double dt,
    const std::vector<double>* q_wall,
    const SourceTerms* sources) const
{
    int N = mesh.N;

    // Compute phasic state and closures
    compute_phasic_state(state, mesh);

    // Save old void fraction and enthalpies
    std::vector<double> alpha_old = state.alpha;
    std::vector<double> h_l_old = state.h_l;
    std::vector<double> h_v_old = state.h_v;

    auto& mdot = state.mdot;

    for (int i = 0; i < N; ++i) {
        auto& is = iface_states_[i];
        auto& cr = closure_results_[i];
        auto& dr = drift_results_[i];

        double rl = is.rho_l;
        double rv = is.rho_v;
        double al = alpha_old[i];

        // ────────────────────────────────────────────────────
        // Void fraction update (from vapor mass equation)
        // ────────────────────────────────────────────────────
        // (αρ_v)^{n+1} = (αρ_v)^n + dt/V * [flux_v_net + V*Γ]

        // Compute phasic fluxes at faces using drift-flux.
        // Face properties are interpolated between adjacent cells;
        // drift-flux parameters (C_0, V_gj) are also face-averaged.

        // Inlet face (between cell i-1 and cell i)
        double alpha_in_face = (i > 0) ? 0.5 * (alpha_old[i - 1] + al)
                                       : bc.alpha_in;
        double rho_l_in = (i > 0) ? 0.5 * (iface_states_[i - 1].rho_l + rl)
                                  : rl;
        double rho_v_in = (i > 0) ? 0.5 * (iface_states_[i - 1].rho_v + rv)
                                  : rv;
        double rho_m_in = (1.0 - alpha_in_face) * rho_l_in
                        + alpha_in_face * rho_v_in;
        // Interpolate drift parameters at inlet face
        double C0_in  = (i > 0) ? 0.5 * (drift_results_[i - 1].C_0  + dr.C_0)  : dr.C_0;
        double Vgj_in = (i > 0) ? 0.5 * (drift_results_[i - 1].V_gj + dr.V_gj) : dr.V_gj;
        auto [mdot_l_in, mdot_v_in] = split_phasic_flux(
            mdot[i], alpha_in_face, rho_l_in, rho_v_in, rho_m_in,
            C0_in, Vgj_in, mesh.A_flow);

        // Outlet face (between cell i and cell i+1)
        double alpha_out_face = (i < N - 1) ? 0.5 * (al + alpha_old[i + 1])
                                            : al;
        double rho_l_out = (i < N - 1) ? 0.5 * (rl + iface_states_[i + 1].rho_l)
                                       : rl;
        double rho_v_out = (i < N - 1) ? 0.5 * (rv + iface_states_[i + 1].rho_v)
                                       : rv;
        double rho_m_out = (1.0 - alpha_out_face) * rho_l_out
                         + alpha_out_face * rho_v_out;
        // Interpolate drift parameters at outlet face
        double C0_out  = (i < N - 1) ? 0.5 * (dr.C_0  + drift_results_[i + 1].C_0)  : dr.C_0;
        double Vgj_out = (i < N - 1) ? 0.5 * (dr.V_gj + drift_results_[i + 1].V_gj) : dr.V_gj;
        auto [mdot_l_out, mdot_v_out] = split_phasic_flux(
            mdot[i + 1], alpha_out_face, rho_l_out, rho_v_out, rho_m_out,
            C0_out, Vgj_out, mesh.A_flow);

        // Vapor mass flux balance
        double flux_v_net = mdot_v_in - mdot_v_out;
        double S_void = (sources && !sources->void_frac.empty())
                       ? sources->void_frac[i] : 0.0;
        double alpha_rho_v_new = al * rv
            + dt / mesh.V * (flux_v_net + mesh.V * (cr.Gamma + S_void));

        // Get new vapor density at new pressure.
        // Floor at 1% of saturation density to avoid division by zero
        // while remaining physically reasonable across pressure range.
        double rv_new = phasic_.rho_vapor(state.p[i], state.h_v[i]);
        double rv_floor = std::max(0.01 * phasic_props_[i].rho_v, 0.01);
        if (rv_new < rv_floor) rv_new = rv_floor;

        double alpha_new = alpha_rho_v_new / rv_new;
        alpha_new = std::clamp(alpha_new, 0.0, 1.0);

        // Nucleation floor: when liquid is superheated (Gamma > 0 from
        // closure), enforce a minimum void fraction. Without this, advective
        // loss washes away the nucleation seed before flashing can grow it.
        if (cr.Gamma > 0.0) {
            alpha_new = std::max(alpha_new, 1e-3);
        }

        state.alpha[i] = alpha_new;

        // ────────────────────────────────────────────────────
        // Phasic enthalpy updates
        // Derived in: derivations/five_eq_phasic_energy.py
        // ────────────────────────────────────────────────────

        // Wall heat split: proportional to wetted fraction
        double q_total = (q_wall != nullptr) ? (*q_wall)[i] : 0.0;
        double q_wall_l = q_total * (1.0 - al);
        double q_wall_v = q_total * al;

        // Pressure work (phasic)
        double dp_dt = (state.p[i] - p_old[i]) / dt;

        // --- Liquid enthalpy ---
        double m_l = (1.0 - al) * rl * mesh.V;
        if (m_l <= 1e-12) {
            // Phase absent: reset to saturation so enthalpy is physical
            // when the phase reappears via condensation.
            state.h_l[i] = phasic_props_[i].h_sat_l;
        } else {
            // Advective flux with reconstruction
            double h_LL = (i >= 2) ? h_l_old[i - 2] : ((i >= 1) ? h_l_old[i - 1] : bc.h_l_in);
            double h_L  = (i >= 1) ? h_l_old[i - 1] : bc.h_l_in;
            double h_R  = h_l_old[i];
            double h_RR = (i < N - 1) ? h_l_old[i + 1] : h_l_old[i];
            double h_face_in = recon.face_value(h_LL, h_L, h_R, h_RR, mdot_l_in);

            double h_LL2 = (i >= 1) ? h_l_old[i - 1] : bc.h_l_in;
            double h_L2  = h_l_old[i];
            double h_R2  = (i < N - 1) ? h_l_old[i + 1] : h_l_old[i];
            double h_RR2 = (i < N - 2) ? h_l_old[i + 2] : h_R2;
            double h_face_out = recon.face_value(h_LL2, h_L2, h_R2, h_RR2, mdot_l_out);

            double flux_l = mdot_l_in * (h_face_in - h_l_old[i])
                          - mdot_l_out * (h_face_out - h_l_old[i]);
            double p_work_l = (1.0 - al) * mesh.V * dp_dt;
            double phase_l = -cr.Gamma * h_l_old[i] * mesh.V;
            double qi_l = cr.q_i_l * mesh.V;
            double S_el = (sources && !sources->energy_l.empty())
                         ? sources->energy_l[i] * mesh.V : 0.0;

            double h_l_new = h_l_old[i]
                + dt / m_l * (flux_l + p_work_l + q_wall_l + qi_l + phase_l + S_el);

            // Enthalpy bounds: prevent non-physical values from explicit
            // overshoot during rapid transients. Liquid enthalpy should stay
            // between a reasonable minimum and the vapor saturation enthalpy.
            constexpr double h_min = 1e4;  // 10 kJ/kg (above ice)
            state.h_l[i] = std::clamp(h_l_new, h_min, phasic_props_[i].h_sat_v);
        }  // end liquid update

        // --- Vapor enthalpy ---
        double m_v = al * rv * mesh.V;
        if (m_v <= 1e-12) {
            // Phase absent: reset to saturation so enthalpy is physical
            // when the phase reappears via evaporation.
            state.h_v[i] = phasic_props_[i].h_sat_v;
        } else {
            double h_LL = (i >= 2) ? h_v_old[i - 2] : ((i >= 1) ? h_v_old[i - 1] : bc.h_v_in);
            double h_L  = (i >= 1) ? h_v_old[i - 1] : bc.h_v_in;
            double h_R  = h_v_old[i];
            double h_RR = (i < N - 1) ? h_v_old[i + 1] : h_v_old[i];
            double h_face_in = recon.face_value(h_LL, h_L, h_R, h_RR, mdot_v_in);

            double h_LL2 = (i >= 1) ? h_v_old[i - 1] : bc.h_v_in;
            double h_L2  = h_v_old[i];
            double h_R2  = (i < N - 1) ? h_v_old[i + 1] : h_v_old[i];
            double h_RR2 = (i < N - 2) ? h_v_old[i + 2] : h_R2;
            double h_face_out = recon.face_value(h_LL2, h_L2, h_R2, h_RR2, mdot_v_out);

            double flux_v = mdot_v_in * (h_face_in - h_v_old[i])
                          - mdot_v_out * (h_face_out - h_v_old[i]);
            double p_work_v = al * mesh.V * dp_dt;
            double phase_v = cr.Gamma * h_v_old[i] * mesh.V;
            double qi_v = cr.q_i_v * mesh.V;
            double S_ev = (sources && !sources->energy_v.empty())
                         ? sources->energy_v[i] * mesh.V : 0.0;

            double h_v_new = h_v_old[i]
                + dt / m_v * (flux_v + p_work_v + q_wall_v + qi_v + phase_v + S_ev);

            // Enthalpy bounds: vapor enthalpy should stay between vapor
            // saturation and a reasonable maximum (~4 MJ/kg at 1 atm).
            // Floor at h_sat_v (not h_sat_l): h_v < h_sat_v is non-physical
            // for bulk vapor and produces invalid IAPWS Region 2 inputs
            // (negative density). Condensation is handled by Gamma, not
            // by h_v undershooting saturation.
            constexpr double h_v_max = 4.0e6;  // 4 MJ/kg
            state.h_v[i] = std::clamp(h_v_new, phasic_props_[i].h_sat_v, h_v_max);
        }  // end vapor update
    }
}

} // namespace opal
