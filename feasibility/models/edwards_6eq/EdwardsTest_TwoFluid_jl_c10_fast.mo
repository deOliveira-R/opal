model EdwardsTest_TwoFluid_jl_c10_fast
  "Edwards blowdown: 6-eq two-fluid + J/L strong feedback, fast flashing (tau_flash=0.005)"
  library.Boundary.ClosedEnd closed_end;
  library.Pipes.Pipe1D_TwoFluid pipe(
    redeclare package Medium = library.Media.Water,
    N=24, L=4.096, D=0.073, f_D=0.02,
    p_init=7e6, h_l_init=986.6e3, h_v_init=2772.6e3, alpha_init=1e-6,
    d_b=3e-4, alpha_nucleation=1e-3,
    mu_l_const=2.8e-4,
    // Jones/Lahey relaxation with strong alpha feedback, fast flashing
    use_relaxation=1, tau_flash=0.005, tau_flash_n=1, tau_flash_DT_ref=40.0,
    tau_cond=0.002, C_tau_alpha=10.0,
    // Henry-Fauske critical flow
    use_critical_flow=true,
    critical_flow_model=2.0,
    C_d=0.61, x_ne=0.14, N_param=0.0, c_floor=10.0);
  library.Boundary.RampedBreak break_bc(
    p_back=101325.0,
    C_d_final=0.61,
    t_open=0.003,
    h_set=986.6e3);
equation
  connect(closed_end.port, pipe.port_a);
  connect(pipe.port_b, break_bc.port);
  // Override pipe's C_d_eff with time-varying value from RampedBreak
  pipe.C_d_eff = break_bc.C_d;
end EdwardsTest_TwoFluid_jl_c10_fast;
