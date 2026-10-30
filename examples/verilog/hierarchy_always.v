module ha(a,b,s,c);
    input wire a,b;
    output wire s,c;
    assign s = a ^ b;
    assign c = a & b;
endmodule

module reg1(clk, d, q);
    input wire clk, d;
    output reg q;
    always @(posedge clk)
        q <= d;
endmodule

module top_module(clk, a, b, s, c, q);
    input wire clk, a, b;
    output wire s,c;
    output reg q;

    ha u_ha(a,b,s,c);
    reg1 u_reg(clk, s, q);
endmodule