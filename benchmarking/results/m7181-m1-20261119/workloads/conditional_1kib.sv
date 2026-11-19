module frontend_conditional_13(
    input logic [13 - 1:0] a, b,
    input logic clk, rst_n, sel,
    output logic [13 - 1:0] y
);
  assign y[0] = sel ? (a[0] ? b[0] : ~b[0]) : (b[0] ? a[0] : 1'b0);
  assign y[1] = sel ? (a[1] ? b[1] : ~b[1]) : (b[1] ? a[1] : 1'b0);
  assign y[2] = sel ? (a[2] ? b[2] : ~b[2]) : (b[2] ? a[2] : 1'b0);
  assign y[3] = sel ? (a[3] ? b[3] : ~b[3]) : (b[3] ? a[3] : 1'b0);
  assign y[4] = sel ? (a[4] ? b[4] : ~b[4]) : (b[4] ? a[4] : 1'b0);
  assign y[5] = sel ? (a[5] ? b[5] : ~b[5]) : (b[5] ? a[5] : 1'b0);
  assign y[6] = sel ? (a[6] ? b[6] : ~b[6]) : (b[6] ? a[6] : 1'b0);
  assign y[7] = sel ? (a[7] ? b[7] : ~b[7]) : (b[7] ? a[7] : 1'b0);
  assign y[8] = sel ? (a[8] ? b[8] : ~b[8]) : (b[8] ? a[8] : 1'b0);
  assign y[9] = sel ? (a[9] ? b[9] : ~b[9]) : (b[9] ? a[9] : 1'b0);
  assign y[10] = sel ? (a[10] ? b[10] : ~b[10]) : (b[10] ? a[10] : 1'b0);
  assign y[11] = sel ? (a[11] ? b[11] : ~b[11]) : (b[11] ? a[11] : 1'b0);
  assign y[12] = sel ? (a[12] ? b[12] : ~b[12]) : (b[12] ? a[12] : 1'b0);
endmodule
