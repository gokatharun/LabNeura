module m #(
    parameter W = 2
)(
    input logic [W-1:0] a,
    input logic [W-1:0] b,
    input logic sel,
    output logic [W:0] sum,
    output logic [3:0] partial
);
    always_comb begin
        sum = a + b;
        partial[3:2] = a;
        partial[1:0] = b;
        partial[0] = sel ? a[0] : b[1];
    end
endmodule
