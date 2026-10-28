
module counter_4bit (
    input  wire       clk,
    output reg  [1:0] count
);
    always @(posedge clk)
        count <= count + 1'b1;

endmodule
