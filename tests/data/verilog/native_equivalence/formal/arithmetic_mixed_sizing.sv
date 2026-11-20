module m(
    input wire signed [1:0] signed_a,
    input wire [3:0] unsigned_b,
    input wire signed [3:0] signed_b,
    input wire [3:0] unsigned_a,
    output wire [3:0] mixed_add,
    output wire signed [3:0] signed_add,
    output wire [3:0] unsigned_sub,
    output wire [4:0] carried_add,
    output wire signed signed_less,
    output wire unsigned_less
);
    assign mixed_add = signed_a + unsigned_b;
    assign signed_add = signed_a + signed_b;
    assign unsigned_sub = unsigned_a - unsigned_b;
    assign carried_add = unsigned_a + unsigned_b;
    assign signed_less = signed_a < signed_b;
    assign unsigned_less = unsigned_a < unsigned_b;
endmodule
