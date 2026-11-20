module m(
    input wire [7:0] data,
    input wire [2:0] index,
    output wire y
);
    assign y = data[index];
endmodule
