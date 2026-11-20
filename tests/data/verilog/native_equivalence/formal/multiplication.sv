module m(
    input wire [3:0] a,
    input wire [3:0] b,
    input wire signed [2:0] signed_a,
    input wire signed [2:0] signed_b,
    output wire [7:0] unsigned_product,
    output wire signed [5:0] signed_product
);
    assign unsigned_product = a * b;
    assign signed_product = signed_a * signed_b;
endmodule
