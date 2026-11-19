package parser_types;
    parameter int DEFAULT_WIDTH = 8;
endpackage

interface stream_if #(parameter int WIDTH = 8) (
    input logic clk,
    input logic [WIDTH-1:0] data
);
    logic ready;
endinterface

module child #(
    parameter int WIDTH = 8,
    parameter HEIGHT = 2
) (
    input logic [WIDTH-1:0] data,
    output logic valid
);
    assign valid = |data;
endmodule

module top (
    input clk,
    input [7:0] data,
    output logic result
);
    logic [7:0] intermediate [0:1];
    wire ready;
    assign intermediate[0] = data + 1;
    assign result = (intermediate[0] * 2) >= 4 ? ready : 1'b0;
    child #(.WIDTH(8)) u_named (.data(data), .valid(ready));
    child #(8) u_positional (data, ready);
endmodule
