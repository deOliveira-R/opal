model EdwardsTest_IAPWS_CritFlow
  "Edwards blowdown with IAPWS + Ransom-Trapp critical flow, N=24"
  library.Boundary.ClosedEnd closed_end;
  library.Pipes.Pipe1D pipe(
    redeclare package Medium = library.Media.Water,
    N=24, L=4.096, D=0.073, f_D=0.02,
    p_init=7e6, h_init=986.6e3,
    use_critical_flow=true,
    C_d=0.87, x_trans=0.10, c_floor=10.0);
  library.Boundary.PressureSource atm(p_set=101325.0, h_set=986.6e3);
equation
  connect(closed_end.port, pipe.port_a);
  connect(pipe.port_b, atm.port);
  pipe.C_d_eff = pipe.C_d;
end EdwardsTest_IAPWS_CritFlow;
