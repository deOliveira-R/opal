model EdwardsTest_IAPWS
  "Edwards blowdown with IAPWS-IF97 properties, N=24"
  library.Boundary.ClosedEnd closed_end;
  library.Pipes.Pipe1D pipe(
    redeclare package Medium = library.Media.Water,
    N=24, L=4.096, D=0.073, f_D=0.02,
    p_init=7e6, h_init=986.6e3);
  library.Boundary.PressureSource atm(p_set=101325.0, h_set=986.6e3);
equation
  connect(closed_end.port, pipe.port_a);
  connect(pipe.port_b, atm.port);
end EdwardsTest_IAPWS;
