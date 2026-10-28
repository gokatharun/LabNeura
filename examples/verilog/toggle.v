module toggle(t,clk,q);
    input wire t,clk;
    output reg q;

    always @(posedge clk)
        q <= q ^ t;
endmodule