module frontend_comment_9(
    input logic [9 - 1:0] a, b,
    input logic clk, rst_n, sel,
    output logic [9 - 1:0] y
);
  // signal 0: deterministic Boolean cone for profiling source trivia
  assign y[0] = a[0] ^ b[0];
  // signal 1: deterministic Boolean cone for profiling source trivia
  assign y[1] = a[1] ^ b[1];
  // signal 2: deterministic Boolean cone for profiling source trivia
  assign y[2] = a[2] ^ b[2];
  // signal 3: deterministic Boolean cone for profiling source trivia
  assign y[3] = a[3] ^ b[3];
  // signal 4: deterministic Boolean cone for profiling source trivia
  assign y[4] = a[4] ^ b[4];
  // signal 5: deterministic Boolean cone for profiling source trivia
  assign y[5] = a[5] ^ b[5];
  // signal 6: deterministic Boolean cone for profiling source trivia
  assign y[6] = a[6] ^ b[6];
  // signal 7: deterministic Boolean cone for profiling source trivia
  assign y[7] = a[7] ^ b[7];
  // signal 8: deterministic Boolean cone for profiling source trivia
  assign y[8] = a[8] ^ b[8];
endmodule
