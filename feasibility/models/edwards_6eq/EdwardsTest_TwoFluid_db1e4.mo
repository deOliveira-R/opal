model EdwardsTest_TwoFluid_db1e4
  "Edwards blowdown: 6-eq two-fluid, d_b=1e-4 (3x stronger HT)"
  library.Boundary.ClosedEnd closed_end;
  library.Pipes.Pipe1D_TwoFluid pipe(
    redeclare package Medium = library.Media.Water,
    N=24, L=4.096, D=0.073, f_D=0.02,
    p_init=7e6, h_l_init=986.6e3, h_v_init=2772.6e3, alpha_init=1e-6,
    d_b=1e-4, alpha_nucleation=1e-3,
    mu_l_const=2.8e-4,
    use_critical_flow=true,
    critical_flow_model=2.0,
    C_d=0.61, x_ne=0.05, N_param=0.0, c_floor=10.0);
  library.Boundary.RampedBreak break_bc(
    p_back=101325.0,
    C_d_final=0.61,
    t_open=0.003,
    h_set=986.6e3);
equation
  connect(closed_end.port, pipe.port_a);
  connect(pipe.port_b, break_bc.port);
  pipe.C_d_eff = break_bc.C_d;
end EdwardsTest_TwoFluid_db1e4;
