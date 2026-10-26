
module mux_4to1 (
    input  wire [3:0] I,
    input  wire [1:0] S,
    output wire       Y
);

assign Y = I[S];

endmodule
