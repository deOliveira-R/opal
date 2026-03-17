#pragma once
/**
 * iapws97.hpp — IAPWS-IF97 property package (C++).
 *
 * Direct port of library/Media/IF97/ Modelica files.
 * Regions 1 (compressed liquid), 2 (superheated steam), 4 (two-phase).
 * All coefficients from IAPWS-IF97 standard tables.
 *
 * Usage: create IAPWSIF97Properties, call evaluate(p, h) → FluidProps.
 */

#include "properties.hpp"
#include <cmath>
#include <cstdio>

namespace opal {

class IAPWSIF97Properties : public FluidProperties {
public:

    // =====================================================================
    // Constants (from IF97/Constants.mo)
    // =====================================================================

    static constexpr double R  = 461.526;    // J/(kg·K)
    static constexpr double Tc = 647.096;    // K
    static constexpr double pc = 22.064e6;   // Pa

    // Region 1 reducing
    static constexpr double p_star_R1 = 16.53e6;  // Pa
    static constexpr double T_star_R1 = 1386.0;   // K

    // Region 2 reducing
    static constexpr double p_star_R2 = 1.0e6;    // Pa
    static constexpr double T_star_R2 = 540.0;    // K

    // =====================================================================
    // Region 1 — 34-term Gibbs function (IAPWS-IF97 Table 2)
    // g(π,τ) = Σ n_i (7.1−π)^I_i (τ−1.222)^J_i
    // =====================================================================

    static constexpr int I_R1[34] = {
        0,0,0,0,0,0,0,0,1,1,1,1,1,1,2,2,2,2,2,3,3,3,4,4,4,5,8,8,21,23,29,30,31,32
    };
    static constexpr int J_R1[34] = {
        -2,-1,0,1,2,3,4,5,-9,-7,-1,0,1,3,-3,0,1,3,17,-4,0,6,-5,-2,10,-8,-11,-6,-29,-31,-38,-39,-40,-41
    };
    static constexpr double n_R1[34] = {
         1.4632971213167e-1,  -8.4548187169114e-1, -3.7563603672040e0,
         3.3855169168385e0,   -9.5791963387872e-1,  1.5772038513228e-1,
        -1.6616417199501e-2,   8.1214629983568e-4,  2.8319080123804e-4,
        -6.0706301565874e-4,  -1.8990068218419e-2, -3.2529748770505e-2,
        -2.1841717175414e-2,  -5.2838357969930e-5, -4.7184321073267e-4,
        -3.0001780793026e-4,   4.7661393906987e-5, -4.4141845330846e-6,
        -7.2694996297594e-16, -3.1679644845054e-5, -2.8270797985312e-6,
        -8.5205128120103e-10, -2.2425281908000e-6, -6.5171222895601e-7,
        -1.4341729937924e-13, -4.0516996860117e-7, -1.2734301741641e-9,
        -1.7424871230634e-10, -6.8762131295531e-19,  1.4478307828521e-20,
         2.6335781662795e-23, -1.1947622640071e-23,  1.8228094581404e-24,
        -9.3537087292458e-26
    };

    // Region 1 Gibbs derivatives
    static double g_pi_R1(double pi, double tau) {
        double s = 0.0;
        double dp = 7.1 - pi;
        double dt = tau - 1.222;
        for (int i = 0; i < 34; ++i)
            s += -n_R1[i] * I_R1[i] * std::pow(dp, I_R1[i] - 1) * std::pow(dt, J_R1[i]);
        return s;
    }
    static double g_pipi_R1(double pi, double tau) {
        double s = 0.0;
        double dp = 7.1 - pi;
        double dt = tau - 1.222;
        for (int i = 0; i < 34; ++i)
            s += n_R1[i] * I_R1[i] * (I_R1[i] - 1) * std::pow(dp, I_R1[i] - 2) * std::pow(dt, J_R1[i]);
        return s;
    }
    static double g_tau_R1(double pi, double tau) {
        double s = 0.0;
        double dp = 7.1 - pi;
        double dt = tau - 1.222;
        for (int i = 0; i < 34; ++i)
            s += n_R1[i] * std::pow(dp, I_R1[i]) * J_R1[i] * std::pow(dt, J_R1[i] - 1);
        return s;
    }
    static double g_tautau_R1(double pi, double tau) {
        double s = 0.0;
        double dp = 7.1 - pi;
        double dt = tau - 1.222;
        for (int i = 0; i < 34; ++i)
            s += n_R1[i] * std::pow(dp, I_R1[i]) * J_R1[i] * (J_R1[i] - 1) * std::pow(dt, J_R1[i] - 2);
        return s;
    }
    static double g_pitau_R1(double pi, double tau) {
        double s = 0.0;
        double dp = 7.1 - pi;
        double dt = tau - 1.222;
        for (int i = 0; i < 34; ++i)
            s += -n_R1[i] * I_R1[i] * std::pow(dp, I_R1[i] - 1) * J_R1[i] * std::pow(dt, J_R1[i] - 1);
        return s;
    }

    // Region 1 properties
    static double h_pT_R1(double p, double T) {
        double pi = p / p_star_R1;
        double tau = T_star_R1 / T;
        return R * T * tau * g_tau_R1(pi, tau);
    }
    static double cp_pT_R1(double p, double T) {
        double pi = p / p_star_R1;
        double tau = T_star_R1 / T;
        return -R * tau * tau * g_tautau_R1(pi, tau);
    }
    static double rho_pT_R1(double p, double T) {
        double pi = p / p_star_R1;
        double tau = T_star_R1 / T;
        double v = (R * T / p) * pi * g_pi_R1(pi, tau);
        return 1.0 / v;
    }

    static double T_ph_R1(double p, double h) {
        // Simple starting guess — mid-range for Region 1 (273–623 K).
        // Newton converges in ~8 iterations from any reasonable guess.
        double T_iter = 400.0;
        for (int iter = 0; iter < 10; ++iter) {
            double f  = h_pT_R1(p, T_iter) - h;
            double df = cp_pT_R1(p, T_iter);
            T_iter -= f / df;
            T_iter = std::max(273.15, std::min(T_iter, 623.15));
        }
        return T_iter;
    }

    // =====================================================================
    // Region 2 — ideal (9-term) + residual (43-term) Gibbs
    // =====================================================================

    // Ideal part (Table 10)
    static constexpr int J0_R2[9] = {0,1,-5,-4,-3,-2,-1,2,3};
    static constexpr double n0_R2[9] = {
        -9.6927686500217e0,   1.0086655968018e1,  -5.6087911283020e-3,
         7.1452738081455e-2, -4.0710498223928e-1,  1.4240819171444e0,
        -4.3839511319450e0,  -2.8408632460772e-1,  2.1268463753307e-2
    };

    // Residual part (Table 11)
    static constexpr int Ir_R2[43] = {
        1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,5,6,6,6,7,7,7,8,8,9,10,10,10,16,16,18,20,20,20,21,22,23,24,24,24
    };
    static constexpr int Jr_R2[43] = {
        0,1,2,3,6,1,2,4,7,36,0,1,3,6,35,1,2,3,7,3,16,35,0,11,25,8,36,13,4,10,14,29,50,57,20,35,48,21,53,39,26,40,58
    };
    static constexpr double nr_R2[43] = {
        -1.7731742473213e-3, -1.7834862292358e-2, -4.5996013696365e-2,
        -5.7581259083432e-2, -5.0325278727930e-2, -3.3032641670203e-5,
        -1.8948987516315e-4, -3.9392777243355e-3, -4.3797295650573e-2,
        -2.6674547914087e-5,  2.0481737692309e-8,  4.3870667284435e-7,
        -3.2277677238570e-5, -1.5033924542148e-3, -4.0668253562649e-2,
        -7.8847309559367e-10, 1.2790717852285e-8,  4.8225372718507e-7,
         2.2922076337661e-6, -1.6714766451061e-11,-2.1171472321355e-3,
        -2.3895741934104e1,  -5.9059564324270e-18,-1.2621808899101e-6,
        -3.8946842435739e-2,  1.1256211360459e-11,-8.2311340897998e0,
         1.9809712802088e-8,  1.0406965210174e-19,-1.0234747095929e-13,
        -1.0018179379511e-9, -8.0882908646985e-11, 1.0693031879409e-1,
        -3.3662250574171e-1,  8.9185845355421e-25, 3.0629316876232e-13,
        -4.2002467698208e-6, -5.9056029685639e-26, 3.7826947613457e-6,
        -1.2768608934681e-15, 7.3087610595061e-29, 5.5414715350778e-17,
        -9.4369707241210e-7
    };

    // Region 2 Gibbs: ideal derivatives
    static double g0_tau_R2(double /*pi*/, double tau) {
        double s = 0.0;
        for (int i = 0; i < 9; ++i)
            s += n0_R2[i] * J0_R2[i] * std::pow(tau, J0_R2[i] - 1);
        return s;
    }
    static double g0_tautau_R2(double /*pi*/, double tau) {
        double s = 0.0;
        for (int i = 0; i < 9; ++i)
            s += n0_R2[i] * J0_R2[i] * (J0_R2[i] - 1) * std::pow(tau, J0_R2[i] - 2);
        return s;
    }

    // Region 2 Gibbs: residual derivatives
    static double gr_pi_R2(double pi, double tau) {
        double s = 0.0;
        double dt = tau - 0.5;
        for (int i = 0; i < 43; ++i)
            s += nr_R2[i] * Ir_R2[i] * std::pow(pi, Ir_R2[i] - 1) * std::pow(dt, Jr_R2[i]);
        return s;
    }
    static double gr_pipi_R2(double pi, double tau) {
        double s = 0.0;
        double dt = tau - 0.5;
        for (int i = 0; i < 43; ++i)
            s += nr_R2[i] * Ir_R2[i] * (Ir_R2[i] - 1) * std::pow(pi, Ir_R2[i] - 2) * std::pow(dt, Jr_R2[i]);
        return s;
    }
    static double gr_tau_R2(double pi, double tau) {
        double s = 0.0;
        double dt = tau - 0.5;
        for (int i = 0; i < 43; ++i)
            s += nr_R2[i] * std::pow(pi, Ir_R2[i]) * Jr_R2[i] * std::pow(dt, Jr_R2[i] - 1);
        return s;
    }
    static double gr_tautau_R2(double pi, double tau) {
        double s = 0.0;
        double dt = tau - 0.5;
        for (int i = 0; i < 43; ++i)
            s += nr_R2[i] * std::pow(pi, Ir_R2[i]) * Jr_R2[i] * (Jr_R2[i] - 1) * std::pow(dt, Jr_R2[i] - 2);
        return s;
    }
    static double gr_pitau_R2(double pi, double tau) {
        double s = 0.0;
        double dt = tau - 0.5;
        for (int i = 0; i < 43; ++i)
            s += nr_R2[i] * Ir_R2[i] * std::pow(pi, Ir_R2[i] - 1) * Jr_R2[i] * std::pow(dt, Jr_R2[i] - 1);
        return s;
    }

    // Region 2 combined derivatives (ideal + residual)
    static double g_pi_tot_R2(double pi, double tau) {
        return 1.0 / pi + gr_pi_R2(pi, tau);
    }
    static double g_pipi_tot_R2(double pi, double tau) {
        return -1.0 / (pi * pi) + gr_pipi_R2(pi, tau);
    }
    static double g_tau_tot_R2(double pi, double tau) {
        return g0_tau_R2(pi, tau) + gr_tau_R2(pi, tau);
    }
    static double g_tautau_tot_R2(double pi, double tau) {
        return g0_tautau_R2(pi, tau) + gr_tautau_R2(pi, tau);
    }
    static double g_pitau_tot_R2(double pi, double tau) {
        return gr_pitau_R2(pi, tau);  // ideal cross-derivative = 0
    }

    // Region 2 properties
    static double h_pT_R2(double p, double T) {
        double pi = p / p_star_R2;
        double tau = T_star_R2 / T;
        return R * T * tau * g_tau_tot_R2(pi, tau);
    }
    static double cp_pT_R2(double p, double T) {
        double pi = p / p_star_R2;
        double tau = T_star_R2 / T;
        return -R * tau * tau * g_tautau_tot_R2(pi, tau);
    }
    static double rho_pT_R2(double p, double T) {
        double pi = p / p_star_R2;
        double tau = T_star_R2 / T;
        double v = (R * T / p) * pi * g_pi_tot_R2(pi, tau);
        return 1.0 / v;
    }

    static double T_ph_R2(double p, double h) {
        // Simple starting guess — mid-range for Region 2 (273–1073 K).
        // Newton converges in ~8 iterations from any reasonable guess.
        double T_iter = 600.0;
        for (int iter = 0; iter < 10; ++iter) {
            double f  = h_pT_R2(p, T_iter) - h;
            double df = cp_pT_R2(p, T_iter);
            T_iter -= f / df;
            T_iter = std::max(273.15, std::min(T_iter, 1073.15));
        }
        return T_iter;
    }

    // =====================================================================
    // Saturation (Region 4)
    // =====================================================================

    static constexpr double n_sat[10] = {
         1.1670521452767e3,  -7.2421316703206e5, -1.7073846940092e1,
         1.2020824702470e4,  -3.2325550322333e6,  1.4915108613530e1,
        -4.8232657361591e3,   4.0511340542057e5, -2.3855557567849e-1,
         6.5017534844798e2
    };

    static double p_sat(double T) {
        double theta = T + n_sat[8] / (T - n_sat[9]);
        double A = theta * theta + n_sat[0] * theta + n_sat[1];
        double B = n_sat[2] * theta * theta + n_sat[3] * theta + n_sat[4];
        double C = n_sat[5] * theta * theta + n_sat[6] * theta + n_sat[7];
        double x = 2.0 * C / (-B + std::sqrt(B * B - 4.0 * A * C));
        return 1.0e6 * x * x * x * x;  // MPa → Pa
    }

    static double T_sat(double p) {
        // Starting guess: Wagner approximation (IAPWS-IF97 Eq. 31)
        double beta = std::pow(p / 1.0e6, 0.25);
        double E = beta * beta + n_sat[2] * beta + n_sat[5];
        double F = n_sat[0] * beta * beta + n_sat[3] * beta + n_sat[6];
        double G = n_sat[1] * beta * beta + n_sat[4] * beta + n_sat[7];
        double D = 2.0 * G / (-F - std::sqrt(F * F - 4.0 * E * G));
        double disc = (n_sat[9] + D) * (n_sat[9] + D) - 4.0 * (n_sat[8] + n_sat[9] * D);
        double T_iter = (n_sat[9] + D - std::sqrt(disc)) / 2.0;

        // Newton: p_sat(T) = p, dp/dT via central difference
        for (int iter = 0; iter < 10; ++iter) {
            double f = p_sat(T_iter) - p;
            double dp_dT = (p_sat(T_iter + 0.005) - p_sat(T_iter - 0.005)) / 0.01;
            T_iter -= f / dp_dT;
        }
        return T_iter;
    }

    // Saturation boundary properties
    static double h_f(double p) { return h_pT_R1(p, T_sat(p)); }
    static double h_g(double p) { return h_pT_R2(p, T_sat(p)); }
    static double h_fg(double p) { return h_g(p) - h_f(p); }
    static double rho_f(double p) { return rho_pT_R1(p, T_sat(p)); }
    static double rho_g(double p) { return rho_pT_R2(p, T_sat(p)); }

    static double quality_ph(double p, double h) {
        return (h - h_f(p)) / h_fg(p);
    }
    static double rho_ph_2phase(double p, double h) {
        double x  = quality_ph(p, h);
        double rf = rho_f(p);
        double rg = rho_g(p);
        return 1.0 / (x / rg + (1.0 - x) / rf);
    }

    // =====================================================================
    // Derivatives (from IF97/Derivatives.mo)
    // (∂ρ/∂p)_h and (∂ρ/∂h)_p via chain rule from Gibbs
    // =====================================================================

    static double drho_dp_h_R1(double p, double T) {
        double pi  = p / p_star_R1;
        double tau = T_star_R1 / T;
        double gpi     = g_pi_R1(pi, tau);
        double gpipi   = g_pipi_R1(pi, tau);
        double gpitau  = g_pitau_R1(pi, tau);
        double gtautau = g_tautau_R1(pi, tau);

        double v     = (R * T / p) * pi * gpi;
        double rho   = 1.0 / v;
        double cp    = -R * tau * tau * gtautau;
        double dv_dp = (R * T / (p * p)) * pi * pi * gpipi;
        double dv_dT = (R * pi / p) * (gpi - tau * gpitau);
        double drho_dT_p = -rho * rho * dv_dT;
        double drho_dp_T = -rho * rho * dv_dp;
        double h_p_val   = (R * T * pi / p) * tau * gpitau;

        return drho_dp_T - drho_dT_p * h_p_val / cp;
    }

    static double drho_dh_p_R1(double p, double T) {
        double pi  = p / p_star_R1;
        double tau = T_star_R1 / T;
        double gpi     = g_pi_R1(pi, tau);
        double gpitau  = g_pitau_R1(pi, tau);
        double gtautau = g_tautau_R1(pi, tau);

        double v     = (R * T / p) * pi * gpi;
        double rho   = 1.0 / v;
        double cp    = -R * tau * tau * gtautau;
        double dv_dT = (R * pi / p) * (gpi - tau * gpitau);
        double drho_dT_p = -rho * rho * dv_dT;

        return drho_dT_p / cp;
    }

    static double drho_dp_h_R2(double p, double T) {
        double pi  = p / p_star_R2;
        double tau = T_star_R2 / T;
        double gpi     = g_pi_tot_R2(pi, tau);
        double gpipi   = g_pipi_tot_R2(pi, tau);
        double gpitau  = g_pitau_tot_R2(pi, tau);
        double gtautau = g_tautau_tot_R2(pi, tau);

        double v     = (R * T / p) * pi * gpi;
        double rho   = 1.0 / v;
        double cp    = -R * tau * tau * gtautau;
        double dv_dp = (R * T / (p * p)) * pi * pi * gpipi;
        double dv_dT = (R * pi / p) * (gpi - tau * gpitau);
        double drho_dT_p = -rho * rho * dv_dT;
        double drho_dp_T = -rho * rho * dv_dp;
        double h_p_val   = (R * T * pi / p) * tau * gpitau;

        return drho_dp_T - drho_dT_p * h_p_val / cp;
    }

    static double drho_dh_p_R2(double p, double T) {
        double pi  = p / p_star_R2;
        double tau = T_star_R2 / T;
        double gpi     = g_pi_tot_R2(pi, tau);
        double gpitau  = g_pitau_tot_R2(pi, tau);
        double gtautau = g_tautau_tot_R2(pi, tau);

        double v     = (R * T / p) * pi * gpi;
        double rho   = 1.0 / v;
        double cp    = -R * tau * tau * gtautau;
        double dv_dT = (R * pi / p) * (gpi - tau * gpitau);
        double drho_dT_p = -rho * rho * dv_dT;

        return drho_dT_p / cp;
    }

    // =====================================================================
    // Region detection (matches Water.mo)
    // =====================================================================

    static int region_ph(double p, double h) {
        if (h < h_f(p)) return 1;
        if (h > h_g(p)) return 2;
        return 4;
    }

    // =====================================================================
    // Unified evaluate() — the FluidProperties interface
    // =====================================================================

    FluidProps evaluate(double p, double h) const override {
        FluidProps fp{};
        int reg = region_ph(p, h);

        if (reg == 1) {
            double T = T_ph_R1(p, h);
            fp.rho       = rho_pT_R1(p, T);
            fp.drho_dp_h = drho_dp_h_R1(p, T);
            fp.drho_dh_p = drho_dh_p_R1(p, T);
            fp.T         = T;
        }
        else if (reg == 2) {
            double T = T_ph_R2(p, h);
            fp.rho       = rho_pT_R2(p, T);
            fp.drho_dp_h = drho_dp_h_R2(p, T);
            fp.drho_dh_p = drho_dh_p_R2(p, T);
            fp.T         = T;
        }
        else {
            // Two-phase (Region 4)
            fp.rho = rho_ph_2phase(p, h);
            fp.T   = T_sat(p);

            // drho_dh_p: analytical (same as Water.mo)
            double rf = rho_f(p);
            double rg = rho_g(p);
            fp.drho_dh_p = -fp.rho * fp.rho * (1.0/rg - 1.0/rf) / h_fg(p);

            // drho_dp_h: finite difference ±500 Pa (same as Water.mo)
            fp.drho_dp_h = (rho_ph_2phase(p + 500.0, h)
                          - rho_ph_2phase(p - 500.0, h)) / 1000.0;
        }

        return fp;
    }
};

} // namespace opal
