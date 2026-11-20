module m(
    input wire [3:0] data,
    input wire [1:0] index,
    output wire y
);
    assign y = data[index + 2'd0];
endmodule
