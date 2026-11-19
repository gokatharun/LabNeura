module frontend_sequential_27(
    input logic [27 - 1:0] a, b,
    input logic clk, rst_n, sel,
    output logic [27 - 1:0] y
);
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) y <= '0;
    else begin
      y[0] <= a[0] ^ b[0];
      y[1] <= a[1] ^ b[1];
      y[2] <= a[2] ^ b[2];
      y[3] <= a[3] ^ b[3];
      y[4] <= a[4] ^ b[4];
      y[5] <= a[5] ^ b[5];
      y[6] <= a[6] ^ b[6];
      y[7] <= a[7] ^ b[7];
      y[8] <= a[8] ^ b[8];
      y[9] <= a[9] ^ b[9];
      y[10] <= a[10] ^ b[10];
      y[11] <= a[11] ^ b[11];
      y[12] <= a[12] ^ b[12];
      y[13] <= a[13] ^ b[13];
      y[14] <= a[14] ^ b[14];
      y[15] <= a[15] ^ b[15];
      y[16] <= a[16] ^ b[16];
      y[17] <= a[17] ^ b[17];
      y[18] <= a[18] ^ b[18];
      y[19] <= a[19] ^ b[19];
      y[20] <= a[20] ^ b[20];
      y[21] <= a[21] ^ b[21];
      y[22] <= a[22] ^ b[22];
      y[23] <= a[23] ^ b[23];
      y[24] <= a[24] ^ b[24];
      y[25] <= a[25] ^ b[25];
      y[26] <= a[26] ^ b[26];
    end
  end
endmodule
