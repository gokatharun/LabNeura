module tb_cpu8080_subset;
    reg clk;
    reg reset;
    wire [7:0] data_in;
    wire [15:0] address;
    wire [7:0] data_out;
    wire mem_read;
    wire mem_write;
    wire halted;
    wire [7:0] accumulator;
    wire [7:0] register_b;

    reg [7:0] program_memory [0:255];

    assign data_in = mem_read ? program_memory[address[7:0]] : 8'h00;

    cpu8080_subset cpu (
        .clk(clk),
        .reset(reset),
        .data_in(data_in),
        .address(address),
        .data_out(data_out),
        .mem_read(mem_read),
        .mem_write(mem_write),
        .halted(halted),
        .accumulator(accumulator),
        .register_b(register_b)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    initial begin
        $readmemh("program.hex", program_memory);
        reset = 1'b1;
        repeat (2) @(posedge clk);
        @(negedge clk);
        reset = 1'b0;

        wait (halted);
        #1;
        if (accumulator !== 8'd5 || register_b !== 8'd3) begin
            $fatal(1, "FAIL: A=%0d B=%0d", accumulator, register_b);
        end
        $display("PASS: MVI A,2; MVI B,3; ADD B; HLT (A=%0d)", accumulator);
        $finish;
    end

    initial begin
        #1000;
        $fatal(1, "FAIL: timeout");
    end
endmodule
