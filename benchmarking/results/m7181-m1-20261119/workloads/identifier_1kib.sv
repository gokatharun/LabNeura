module frontend_identifier_6(
    input logic [6 - 1:0] a, b,
    input logic clk, rst_n, sel,
    output logic [6 - 1:0] y
);
  logic deterministic_intermediate_signal_0000000;
  logic deterministic_intermediate_signal_0000001;
  logic deterministic_intermediate_signal_0000002;
  logic deterministic_intermediate_signal_0000003;
  logic deterministic_intermediate_signal_0000004;
  logic deterministic_intermediate_signal_0000005;
  assign deterministic_intermediate_signal_0000000 = a[0] ^ b[0];
  assign y[0] = deterministic_intermediate_signal_0000000;
  assign deterministic_intermediate_signal_0000001 = a[1] ^ b[1];
  assign y[1] = deterministic_intermediate_signal_0000001;
  assign deterministic_intermediate_signal_0000002 = a[2] ^ b[2];
  assign y[2] = deterministic_intermediate_signal_0000002;
  assign deterministic_intermediate_signal_0000003 = a[3] ^ b[3];
  assign y[3] = deterministic_intermediate_signal_0000003;
  assign deterministic_intermediate_signal_0000004 = a[4] ^ b[4];
  assign y[4] = deterministic_intermediate_signal_0000004;
  assign deterministic_intermediate_signal_0000005 = a[5] ^ b[5];
  assign y[5] = deterministic_intermediate_signal_0000005;
endmodule
