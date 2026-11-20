module m(
    input wire [3:0] a,
    input wire [2:0] amount,
    input wire signed [3:0] signed_a,
    output wire [3:0] quotient,
    output wire [3:0] remainder,
    output wire [3:0] left_shift,
    output wire signed [3:0] arithmetic_right_shift
);
    assign quotient = a / 4'd3;
    assign remainder = a % 4'd3;
    assign left_shift = a << amount;
    assign arithmetic_right_shift = signed_a >>> amount;
endmodule
