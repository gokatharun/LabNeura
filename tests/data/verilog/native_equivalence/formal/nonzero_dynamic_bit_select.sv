module m(
    input wire [7:4] data,
    input wire [1:0] index,
    output wire y
);
    wire [2:0] selected_index;
    assign selected_index = {1'b1, index};
    assign y = data[selected_index];
endmodule
