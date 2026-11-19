module alu_slice #(
    parameter int WIDTH = 16
) (
    input  logic                 clk,
    input  logic [WIDTH-1:0]     lhs,
    input  logic [WIDTH-1:0]     rhs,
    input  logic [1:0]           operation,
    output logic [WIDTH-1:0]     result
);
    // Comment trivia is intentionally included in Slang's token differential.
    logic [WIDTH:0] extended_sum;
    localparam real SCALE = 1.;
    localparam string LABEL = "ALU \" slice";

    assign extended_sum = {1'b0, lhs} + {1'b0, rhs};

    /* Assertion operators are part of the ordinary token stream. */
    property request_acknowledged;
        @(posedge clk) lhs != '0 |-> ##[1:3] rhs != '0;
    endproperty

    always_comb begin
        unique case (operation)
            2'b00: result = lhs & rhs;
            2'b01: result = lhs | rhs;
            2'b10: result = lhs ^ rhs;
            default: result = extended_sum[WIDTH-1:0];
        endcase
    end
endmodule
