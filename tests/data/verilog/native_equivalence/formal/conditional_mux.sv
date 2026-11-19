module m(input logic [1:0] a, b, input logic sel,
         output logic [1:0] y);
  always_comb begin
    if (sel) y = a;
    else y = ~b;
  end
endmodule
