module frontend_parameter_15 #(parameter int WIDTH = 15)(
    input logic [WIDTH - 1:0] a, b,
    input logic clk, rst_n, sel,
    output logic [WIDTH - 1:0] y
);
  localparam logic P_0 = 1'b1;
  localparam logic P_1 = 1'b0;
  localparam logic P_2 = 1'b1;
  localparam logic P_3 = 1'b0;
  localparam logic P_4 = 1'b1;
  localparam logic P_5 = 1'b0;
  localparam logic P_6 = 1'b1;
  localparam logic P_7 = 1'b0;
  localparam logic P_8 = 1'b1;
  localparam logic P_9 = 1'b0;
  localparam logic P_10 = 1'b1;
  localparam logic P_11 = 1'b0;
  localparam logic P_12 = 1'b1;
  localparam logic P_13 = 1'b0;
  localparam logic P_14 = 1'b1;
  assign y[0] = a[0] ^ P_0;
  assign y[1] = a[1] ^ P_1;
  assign y[2] = a[2] ^ P_2;
  assign y[3] = a[3] ^ P_3;
  assign y[4] = a[4] ^ P_4;
  assign y[5] = a[5] ^ P_5;
  assign y[6] = a[6] ^ P_6;
  assign y[7] = a[7] ^ P_7;
  assign y[8] = a[8] ^ P_8;
  assign y[9] = a[9] ^ P_9;
  assign y[10] = a[10] ^ P_10;
  assign y[11] = a[11] ^ P_11;
  assign y[12] = a[12] ^ P_12;
  assign y[13] = a[13] ^ P_13;
  assign y[14] = a[14] ^ P_14;
endmodule
