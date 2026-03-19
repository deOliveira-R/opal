/**
 * hem_model.cpp — HEM (3-equation) FlowModel implementation.
 *
 * Extracted directly from Phase 2 solver.cpp. The physics is identical;
 * the code is restructured to fit the FlowModel interface.
 */

#include "hem_model.hpp"
#include <cmath>

namespace opal {

// ---------------------------------------------------------------------------
// State management
// ---------------------------------------------------------------------------

SolverState HEMModel::make_state(
    const std::vector<double>& p,
    const std::vector<double>& h,
    const std::vector<double>& mdot) const
{
    SolverState s;
    s.p    = p;
    s.h_l  = h;    // HEM uses h_l as mixture enthalpy
    s.mdot = mdot;
    return s;
}

void HEMModel::pack_state(
    const SolverState& state,
    std::vector<double>& out) const
{
    int N = static_cast<int>(state.p.size());
    out.clear();
    out.reserve(3 * N + 1);
    out.insert(out.end(), state.p.begin(), state.p.end());
    out.insert(out.end(), state.h_l.begin(), state.h_l.end());
    out.insert(out.end(), state.mdot.begin(), state.mdot.end());
}

SolverState HEMModel::unpack_state(const double* data, int N) const
{
    SolverState s;
    s.p.assign(data, data + N);
    s.h_l.assign(data + N, data + 2 * N);
    s.mdot.assign(data + 2 * N, data + 3 * N + 1);
    return s;
}

// ---------------------------------------------------------------------------
// Property evaluation
// ---------------------------------------------------------------------------

void HEMModel::evaluate_properties(
    const SolverState& state,
    const FluidPackage& fluid,
    std::vector<FluidProps>& props) const
{
    int N = static_cast<int>(state.p.size());
    props.resize(N);
    for (int i = 0; i < N; ++i) {
        props[i] = fluid.evaluate(state.p[i], state.h_l[i]);
    }
}

// ---------------------------------------------------------------------------
// Face resistance: R = f_D * dx / (2 * D_h * A^2 * rho_face)
// ---------------------------------------------------------------------------

void HEMModel::compute_face_resistance(
    const SolverState& /*state*/,
    double p_in,
    const FaceTransportBC& tbc_in,
    const FluidPackage& fluid,
    const MeshParams& mesh,
    const std::vector<FluidProps>& props,
    std::vector<double>& R_face) const
{
    int N = mesh.N;
    R_face.resize(N + 1);

    double geom = mesh.f_D * mesh.dx / (2.0 * mesh.D_h * mesh.A_flow * mesh.A_flow);

    double h_in = (tbc_in.h_mix != 0.0) ? tbc_in.h_mix : tbc_in.h_l;
    double rho_in = fluid.evaluate(p_in, h_in).rho;
    R_face[0] = geom / (0.5 * (rho_in + props[0].rho));

    for (int i = 1; i < N; ++i) {
        R_face[i] = geom / (0.5 * (props[i - 1].rho + props[i].rho));
    }

    R_face[N] = geom / props[N - 1].rho;
}

void HEMModel::assemble_pressure_system(
    const SolverState& state,
    double p_in, double p_out,
    const MeshParams& mesh,
    const std::vector<FluidProps>& props,
    const std::vector<double>& R_face,
    double dt,
    TridiagCoeffs& tri) const
{
    int N = mesh.N;
    tri.resize(N);

    for (int i = 0; i < N; ++i) {
        double alpha_coeff = mesh.V * props[i].drho_dp_h / dt;
        double inv_R_left  = 1.0 / R_face[i];
        double inv_R_right = 1.0 / R_face[i + 1];

        tri.a[i] = (i > 0)     ? -inv_R_left  : 0.0;
        tri.c[i] = (i < N - 1) ? -inv_R_right : 0.0;
        tri.b[i] = alpha_coeff + inv_R_left + inv_R_right;
        tri.d[i] = alpha_coeff * state.p[i];

        if (i == 0)     tri.d[i] += p_in  * inv_R_left;
        if (i == N - 1) tri.d[i] += p_out * inv_R_right;
    }
}

void HEMModel::update_velocities(
    SolverState& state,
    double p_in, double p_out,
    const MeshParams& mesh,
    const std::vector<double>& R_face) const
{
    int N = mesh.N;
    state.mdot[0] = (p_in - state.p[0]) / R_face[0];
    for (int i = 1; i < N; ++i) {
        state.mdot[i] = (state.p[i - 1] - state.p[i]) / R_face[i];
    }
    state.mdot[N] = (state.p[N - 1] - p_out) / R_face[N];
}

// ---------------------------------------------------------------------------
// Explicit enthalpy update (donor-cell/MUSCL, forward Euler)
// ---------------------------------------------------------------------------

void HEMModel::update_transport(
    SolverState& state,
    const std::vector<double>& p_old,
    const FaceTransportBC& tbc_in,
    const MeshParams& mesh,
    const std::vector<FluidProps>& props,
    const FaceReconstruction& recon,
    double dt,
    const std::vector<double>* q_wall,
    const SolverNumerics& /*numerics*/,
    const SourceTerms* sources) const
{
    int N = mesh.N;
    auto& h = state.h_l;  // HEM: h_l is mixture enthalpy
    auto& mdot = state.mdot;

    // Freeze enthalpy so MUSCL reconstruction reads consistent old values
    std::vector<double> h_old(h);

    for (int i = 0; i < N; ++i) {
        double rho_i = props[i].rho;

        // Inlet face (face i): Dirichlet for inflow, reconstruction for interior
        int ng = recon.ghost_cells();
        double h_face_in;
        if (i == 0 && mdot[i] >= 0.0) {
            h_face_in = tbc_in.h_l;
        } else {
            double h_LL_in, h_L_in, h_R_in, h_RR_in;
            build_stencil(h_old.data(), N, i,
                          tbc_in.h_l, h_old[N - 1], ng,
                          h_LL_in, h_L_in, h_R_in, h_RR_in);
            h_face_in = recon.face_value(h_LL_in, h_L_in, h_R_in, h_RR_in, mdot[i]);
        }

        // Outlet face (face i+1): upwind for outflow, reconstruction for interior
        double h_face_out;
        if (i == N - 1 && mdot[i + 1] >= 0.0) {
            h_face_out = h_old[N - 1];
        } else {
            double h_LL_out, h_L_out, h_R_out, h_RR_out;
            build_stencil(h_old.data(), N, i + 1,
                          tbc_in.h_l, h_old[N - 1], ng,
                          h_LL_out, h_L_out, h_R_out, h_RR_out);
            h_face_out = recon.face_value(h_LL_out, h_L_out, h_R_out, h_RR_out, mdot[i + 1]);
        }

        double flux = mdot[i] * (h_face_in - h_old[i])
                    - mdot[i + 1] * (h_face_out - h_old[i]);

        double p_work = mesh.V * (state.p[i] - p_old[i]) / dt;
        double q = (q_wall != nullptr) ? (*q_wall)[i] : 0.0;
        double S_e = (sources && !sources->energy_l.empty())
                    ? sources->energy_l[i] * mesh.V : 0.0;

        h[i] = h_old[i] + dt / (rho_i * mesh.V) * (flux + p_work + q + S_e);
    }
}

} // namespace opal
