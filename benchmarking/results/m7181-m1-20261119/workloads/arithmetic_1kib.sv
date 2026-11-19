module frontend_arithmetic_18(
    input logic [18 - 1:0] a, b,
    input logic clk, rst_n, sel,
    output logic [18 - 1:0] y
);
  assign y[0] = (a[0] + b[0]) - (a[0] & b[0]);
  assign y[1] = (a[1] + b[1]) - (a[1] & b[1]);
  assign y[2] = (a[2] + b[2]) - (a[2] & b[2]);
  assign y[3] = (a[3] + b[3]) - (a[3] & b[3]);
  assign y[4] = (a[4] + b[4]) - (a[4] & b[4]);
  assign y[5] = (a[5] + b[5]) - (a[5] & b[5]);
  assign y[6] = (a[6] + b[6]) - (a[6] & b[6]);
  assign y[7] = (a[7] + b[7]) - (a[7] & b[7]);
  assign y[8] = (a[8] + b[8]) - (a[8] & b[8]);
  assign y[9] = (a[9] + b[9]) - (a[9] & b[9]);
  assign y[10] = (a[10] + b[10]) - (a[10] & b[10]);
  assign y[11] = (a[11] + b[11]) - (a[11] & b[11]);
  assign y[12] = (a[12] + b[12]) - (a[12] & b[12]);
  assign y[13] = (a[13] + b[13]) - (a[13] & b[13]);
  assign y[14] = (a[14] + b[14]) - (a[14] & b[14]);
  assign y[15] = (a[15] + b[15]) - (a[15] & b[15]);
  assign y[16] = (a[16] + b[16]) - (a[16] & b[16]);
  assign y[17] = (a[17] + b[17]) - (a[17] & b[17]);
endmodule
