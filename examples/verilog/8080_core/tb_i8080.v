module tb_i8080;
    reg clk;
    reg reset;

    wire io;
    wire [15:0] address;
    reg [7:0] data_to_cpu;
    wire [7:0] data_from_cpu;
    wire read;
    wire write;
    wire interrupt_ack;
    reg ready;

    reg [7:0] memory [0:65535];
    integer index;

    i8080 cpu (
        .i_rst(reset),
        .i_clk(clk),
        .i_ready(ready),
        .o_io(io),
        .o_addr(address),
        .i_dat_r(data_to_cpu),
        .o_read(read),
        .o_dat_w(data_from_cpu),
        .o_write(write),
        .i_int(1'b0),
        .o_inta(interrupt_ack)
    );

    always #5 clk = ~clk;

    always @(posedge clk) begin
        if (reset) begin
            ready <= 1'b0;
            data_to_cpu <= 8'h00;
        end else begin
            ready <= read | write;
            if (write && !io)
                memory[address] <= data_from_cpu;
            else
                data_to_cpu <= memory[address];
        end
    end

    initial begin
        clk = 1'b0;
        reset = 1'b1;
        for (index = 0; index < 65536; index = index + 1)
            memory[index] = 8'h00;
        ready = 1'b0;
        $readmemh("program.hex", memory, 0, 45);

        repeat (3) @(posedge clk);
        @(negedge clk);
        reset = 1'b0;

        wait (cpu.op_halt === 1'b1);
        #1;
        if (cpu.r_a !== 8'h43)
            $fatal(1, "A expected 43, got %02x", cpu.r_a);
        if (cpu.r_bc !== 16'h1234)
            $fatal(1, "BC expected 1234, got %04x", cpu.r_bc);
        if (cpu.r_de !== 16'h9942 || cpu.r_hl !== 16'h9942)
            $fatal(1, "DE/HL expected 9942, got DE=%04x HL=%04x",
                   cpu.r_de, cpu.r_hl);
        if (cpu.r_sp !== 16'hF000)
            $fatal(1, "SP expected F000, got %04x", cpu.r_sp);
        if (cpu.r_f_c !== 1'b1)
            $fatal(1, "Carry should remain set after INR");
        if (memory[16'h2000] !== 8'h42 ||
            memory[16'h2001] !== 8'h99 ||
            memory[16'h2002] !== 8'h43)
            $fatal(1, "Unexpected memory results");

        $display("PASS: 8080 instruction and memory checks");
        $finish;
    end

    initial begin
        #20000;
        $fatal(1, "FAIL: CPU did not reach HLT");
    end
endmodule
