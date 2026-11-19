module m(
    input logic [1:0] a,
    input logic [1:0] b,
    input logic [1:0] unsigned_a,
    input logic [1:0] unsigned_b,
    input logic signed [1:0] signed_a,
    input logic signed [1:0] signed_b,
    output logic [1:0] bitwise_and,
    output logic logical_and,
    output logic equal,
    output logic unsigned_less,
    output logic signed_less
);
    assign bitwise_and = a & b;
    assign logical_and = (a && b);
    assign equal = (a == b);
    assign unsigned_less = unsigned_a < unsigned_b;
    assign signed_less = signed_a < signed_b;
endmodule
