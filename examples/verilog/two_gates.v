module two_gates (a, b, c, y);
    input wire a, b, c;
    output wire y;
    wire w;

    and AND1 (w, a, b);
    or OR1 (y, w, c);
endmodule
