// Output is high when at least two of the three inputs are high.
module majority3 (a, b, c, y);
    input wire a, b, c;
    output wire y;
    wire ab, ac, bc, pair_or;

    and AND_AB (ab, a, b);
    and AND_AC (ac, a, c);
    and AND_BC (bc, b, c);
    or OR_PAIRS (pair_or, ab, ac);
    or OR_OUTPUT (y, pair_or, bc);
endmodule
