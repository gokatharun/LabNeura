module frontend_wide_29(
    input logic [29 - 1:0] a, b,
    input logic clk, rst_n, sel,
    output logic [29 - 1:0] y
);
  assign y[0] = a[0] ^ b[3];
  assign y[1] = a[1] ^ b[20];
  assign y[2] = a[2] ^ b[8];
  assign y[3] = a[3] ^ b[25];
  assign y[4] = a[4] ^ b[13];
  assign y[5] = a[5] ^ b[1];
  assign y[6] = a[6] ^ b[18];
  assign y[7] = a[7] ^ b[6];
  assign y[8] = a[8] ^ b[23];
  assign y[9] = a[9] ^ b[11];
  assign y[10] = a[10] ^ b[28];
  assign y[11] = a[11] ^ b[16];
  assign y[12] = a[12] ^ b[4];
  assign y[13] = a[13] ^ b[21];
  assign y[14] = a[14] ^ b[9];
  assign y[15] = a[15] ^ b[26];
  assign y[16] = a[16] ^ b[14];
  assign y[17] = a[17] ^ b[2];
  assign y[18] = a[18] ^ b[19];
  assign y[19] = a[19] ^ b[7];
  assign y[20] = a[20] ^ b[24];
  assign y[21] = a[21] ^ b[12];
  assign y[22] = a[22] ^ b[0];
  assign y[23] = a[23] ^ b[17];
  assign y[24] = a[24] ^ b[5];
  assign y[25] = a[25] ^ b[22];
  assign y[26] = a[26] ^ b[10];
  assign y[27] = a[27] ^ b[27];
  assign y[28] = a[28] ^ b[15];
endmodule
