module m(
    input logic [2:0] a,
    input logic [2:0] b,
    output logic [2:0] sum,
    output logic [2:0] difference
);
    assign sum = a + b;
    assign difference = a - b;
endmodule
