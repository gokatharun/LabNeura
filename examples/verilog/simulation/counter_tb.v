
module counter_tb;

reg clk;
reg rst;
wire [3:0] count;

counter dut (
    .clk(clk),
    .rst(rst),
    .count(count)
);

// Clock generation
always #5 clk = ~clk;

// Test sequence
initial begin

    $dumpfile("counter.vcd");
    $dumpvars(0, counter_tb);

    clk = 0;
    rst = 1;

    #10;
    rst = 0;

    repeat (20) begin
        @(posedge clk);
        #1;
        $display("Time=%0t Count=%d", $time, count);
    end

    $finish;
end

endmodule
