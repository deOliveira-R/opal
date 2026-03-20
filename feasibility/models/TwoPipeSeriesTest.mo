model TwoPipeSeriesTest
  "Two pipes connected in series — feasibility test for multi-component extraction"
  library.Boundary.PressureSource inlet(p_set=10e6, h_set=800e3);
  library.Pipes.Pipe1D pipe1(
    redeclare package Medium = library.Media.SimpleFluid,
    N=3, L=3.0, D=0.1, f_D=0.02,
    p_init=10e6, h_init=800e3);
  library.Pipes.Pipe1D pipe2(
    redeclare package Medium = library.Media.SimpleFluid,
    N=3, L=3.0, D=0.1, f_D=0.02,
    p_init=10e6, h_init=800e3);
  library.Boundary.PressureSource outlet(p_set=9.5e6, h_set=800e3);
equation
  connect(inlet.port, pipe1.port_a);
  connect(pipe1.port_b, pipe2.port_a);
  connect(pipe2.port_b, outlet.port);
  pipe1.C_d_eff = pipe1.C_d;
  pipe2.C_d_eff = pipe2.C_d;
end TwoPipeSeriesTest;
