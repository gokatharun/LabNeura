module m(input logic a, b, sel,
         output logic y_and, y_or, y_xor, y_not, y_mux, y_hi, y_lo);
  assign y_and = a & b;
  assign y_or = a | b;
  assign y_xor = a ^ b;
  assign y_not = ~a;
  assign y_mux = sel ? a : b;
  assign y_hi = 1'b1;
  assign y_lo = 1'b0;
endmodule
