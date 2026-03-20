model EdwardsTest_DriftFlux
  "Edwards blowdown: 5-eq drift-flux + IAPWS + Ransom-Trapp + two-phase friction"
  library.Boundary.ClosedEnd closed_end;
  library.Pipes.Pipe1D_DriftFlux pipe(
    redeclare package Medium = library.Media.Water,
    N=24, L=4.096, D=0.073, f_D=0.02,
    p_init=7e6, h_l_init=986.6e3, h_v_init=2772.6e3, alpha_init=1e-6,
    H_i=1e7, C_0=1.0, alpha_nucleation=1e-3,
    use_critical_flow=true, C_d=0.87, x_trans=0.10, c_floor=10.0,
    use_two_phase_friction=true);
  library.Boundary.PressureSource atm(p_set=101325.0, h_set=986.6e3);
equation
  connect(closed_end.port, pipe.port_a);
  connect(pipe.port_b, atm.port);
end EdwardsTest_DriftFlux;
