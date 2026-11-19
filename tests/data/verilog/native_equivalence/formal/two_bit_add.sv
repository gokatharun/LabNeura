module m(input logic [1:0] a, b, output logic [2:0] y);
  assign y = {1'b0, a} + {1'b0, b};
endmodule
