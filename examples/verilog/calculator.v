module calculator (
    input clk,
    input [1:0] op,
    input [3:0] a,
    input [3:0] b,
    output reg [4:0] result
);
    wire [4:0] a5 = {1'b0, a};
    wire [4:0] b5 = {1'b0, b};

    always @(posedge clk) begin
        case (op)
            2'b00: result <= a5 + b5;
            2'b01: result <= a5 - b5;
            2'b10: result <= {1'b0, a & b};
            2'b11: result <= {1'b0, a | b};
        endcase
    end
endmodule