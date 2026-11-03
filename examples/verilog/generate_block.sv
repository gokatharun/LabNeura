module generate_process_test (
    input  logic       clk,
    input  logic       rst,
    input  logic [3:0] a,
    input  logic [3:0] b,
    output logic [3:0] q
);
    wire [3:0] xor_bits;

    for (genvar i = 0; i < 4; i++) begin : gen_xor
        assign xor_bits[i] = a[i] ^ b[i];
    end

    always_ff @(posedge clk or posedge rst) begin
        if (rst)
            q <= 4'b0000;
        else
            q <= xor_bits;
    end
endmodule