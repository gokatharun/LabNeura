module m(
    input logic [3:0] descending,
    output logic [0:3] ascending,
    output logic [2:0] selected,
    output logic [4:0] concatenated,
    output logic [3:0] replicated
);
    assign ascending = descending;
    assign selected = descending[3:1];
    assign concatenated = {descending[0], descending[3:1], descending[0]};
    assign replicated = {2{descending[1:0]}};
endmodule
