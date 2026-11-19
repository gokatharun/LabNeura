module frontend_case_15(
    input logic [15 - 1:0] a, b,
    input logic clk, rst_n, sel,
    output logic [15 - 1:0] y
);
  always_comb begin
    case ({a[0], b[0]})
      2'b00: y = {1'b0, 1'b0, 1'b0, 1'b0, 1'b0, 1'b0, 1'b0, 1'b0, 1'b0, 1'b0, 1'b0, 1'b0, 1'b0, 1'b0, 1'b0};
      2'b01: y = {(sel ? a[14] : b[14]), (sel ? a[13] : b[13]), (sel ? a[12] : b[12]), (sel ? a[11] : b[11]), (sel ? a[10] : b[10]), (sel ? a[9] : b[9]), (sel ? a[8] : b[8]), (sel ? a[7] : b[7]), (sel ? a[6] : b[6]), (sel ? a[5] : b[5]), (sel ? a[4] : b[4]), (sel ? a[3] : b[3]), (sel ? a[2] : b[2]), (sel ? a[1] : b[1]), (sel ? a[0] : b[0])};
      2'b10: y = {~(a[14] ^ b[14]), ~(a[13] ^ b[13]), ~(a[12] ^ b[12]), ~(a[11] ^ b[11]), ~(a[10] ^ b[10]), ~(a[9] ^ b[9]), ~(a[8] ^ b[8]), ~(a[7] ^ b[7]), ~(a[6] ^ b[6]), ~(a[5] ^ b[5]), ~(a[4] ^ b[4]), ~(a[3] ^ b[3]), ~(a[2] ^ b[2]), ~(a[1] ^ b[1]), ~(a[0] ^ b[0])};
      default: y = {1'b1, 1'b1, 1'b1, 1'b1, 1'b1, 1'b1, 1'b1, 1'b1, 1'b1, 1'b1, 1'b1, 1'b1, 1'b1, 1'b1, 1'b1};
    endcase
  end
endmodule
