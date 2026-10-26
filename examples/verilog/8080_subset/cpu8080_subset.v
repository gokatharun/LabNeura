// Educational 8080-style CPU subset, not a complete Intel 8080.
module cpu8080_subset (
    input wire clk,
    input wire reset,
    input wire [7:0] data_in,
    output reg [15:0] address,
    output reg [7:0] data_out,
    output reg mem_read,
    output reg mem_write,
    output reg halted,
    output reg [7:0] accumulator,
    output reg [7:0] register_b
);
    localparam FETCH = 3'd0;
    localparam DECODE = 3'd1;
    localparam LOAD_ACCUMULATOR = 3'd2;
    localparam LOAD_B = 3'd3;
    localparam HALT = 3'd4;

    localparam NOP = 8'h00;
    localparam MVI_B = 8'h06;
    localparam MVI_A = 8'h3e;
    localparam MOV_A_B = 8'h78;
    localparam ADD_B = 8'h80;
    localparam HLT = 8'h76;

    reg [2:0] state;
    reg [15:0] program_counter;
    reg [7:0] instruction;

    always @* begin
        address = program_counter;
        data_out = 8'h00;
        mem_read = 1'b0;
        mem_write = 1'b0;

        if ((state == FETCH) ||
            (state == LOAD_ACCUMULATOR) ||
            (state == LOAD_B)) begin
            mem_read = 1'b1;
        end
    end

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            state <= FETCH;
            program_counter <= 16'h0000;
            instruction <= NOP;
            accumulator <= 8'h00;
            register_b <= 8'h00;
            halted <= 1'b0;
        end else begin
            case (state)
                FETCH: begin
                    instruction <= data_in;
                    program_counter <= program_counter + 16'h0001;
                    state <= DECODE;
                end
                DECODE: begin
                    case (instruction)
                        NOP: state <= FETCH;
                        MVI_A: state <= LOAD_ACCUMULATOR;
                        MVI_B: state <= LOAD_B;
                        MOV_A_B: begin
                            accumulator <= register_b;
                            state <= FETCH;
                        end
                        ADD_B: begin
                            accumulator <= accumulator + register_b;
                            state <= FETCH;
                        end
                        HLT: begin
                            halted <= 1'b1;
                            state <= HALT;
                        end
                        default: begin
                            halted <= 1'b1;
                            state <= HALT;
                        end
                    endcase
                end
                LOAD_ACCUMULATOR: begin
                    accumulator <= data_in;
                    program_counter <= program_counter + 16'h0001;
                    state <= FETCH;
                end
                LOAD_B: begin
                    register_b <= data_in;
                    program_counter <= program_counter + 16'h0001;
                    state <= FETCH;
                end
                HALT: state <= HALT;
                default: begin
                    halted <= 1'b1;
                    state <= HALT;
                end
            endcase
        end
    end
endmodule
