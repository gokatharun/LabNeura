module frontend_literal_24(
    input logic [24 - 1:0] a, b,
    input logic clk, rst_n, sel,
    output logic [24 - 1:0] y
);
  assign y[0] = a[0] ^ 32'h00000000;
  assign y[1] = a[1] ^ 32'h9e3779b1;
  assign y[2] = a[2] ^ 32'h3c6ef362;
  assign y[3] = a[3] ^ 32'hdaa66d13;
  assign y[4] = a[4] ^ 32'h78dde6c4;
  assign y[5] = a[5] ^ 32'h17156075;
  assign y[6] = a[6] ^ 32'hb54cda26;
  assign y[7] = a[7] ^ 32'h538453d7;
  assign y[8] = a[8] ^ 32'hf1bbcd88;
  assign y[9] = a[9] ^ 32'h8ff34739;
  assign y[10] = a[10] ^ 32'h2e2ac0ea;
  assign y[11] = a[11] ^ 32'hcc623a9b;
  assign y[12] = a[12] ^ 32'h6a99b44c;
  assign y[13] = a[13] ^ 32'h08d12dfd;
  assign y[14] = a[14] ^ 32'ha708a7ae;
  assign y[15] = a[15] ^ 32'h4540215f;
  assign y[16] = a[16] ^ 32'he3779b10;
  assign y[17] = a[17] ^ 32'h81af14c1;
  assign y[18] = a[18] ^ 32'h1fe68e72;
  assign y[19] = a[19] ^ 32'hbe1e0823;
  assign y[20] = a[20] ^ 32'h5c5581d4;
  assign y[21] = a[21] ^ 32'hfa8cfb85;
  assign y[22] = a[22] ^ 32'h98c47536;
  assign y[23] = a[23] ^ 32'h36fbeee7;
endmodule
