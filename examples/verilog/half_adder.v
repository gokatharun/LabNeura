module half_adder(a,b,s,c);
    input wire a,b;
    output wire s,c;

    xor sum(s,a,b);
    and carry(c,a,b);
endmodule