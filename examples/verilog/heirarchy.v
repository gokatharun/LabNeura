module ha1(a,b,s,c);
    input wire a,b;
    output wire s,c;

    assign s = a ^ b;
    assign c = a & b;
endmodule;

module ha2(a,b,s,c);
    input wire a,b;
    output wire s,c;

    assign s = a ^ b;
    assign c = a & b;
endmodule;

module top_module(a,b,s1,c1,s2,c2);
    input wire a,b;
    output wire s1,c1,s2,c2;

    ha1 u_ha1(a,b,s1,c1);
    ha2 u_ha2(a,b,s2,c2);
endmodule;