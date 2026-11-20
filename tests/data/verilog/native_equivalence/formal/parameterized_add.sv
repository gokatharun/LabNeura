module m #(
    parameter W = 3
) (
    input wire [W-1:0] a,
    input wire [W-1:0] b,
    output wire [W:0] y
);
    assign y = a + b;
endmodule
