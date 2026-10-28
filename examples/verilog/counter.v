
module counter_4bit (
    input  wire       clk,
    output reg  [4:0] count
);
    always @(posedge clk)
        count <= count + 1;

endmodule
