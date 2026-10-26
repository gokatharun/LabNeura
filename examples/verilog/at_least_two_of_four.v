// Output is high when any pair of the four inputs is high.
module at_least_two_of_four (a, b, c, d, y);
    input wire a, b, c, d;
    output wire y;
    wire ab, ac, ad, bc, bd, cd;
    wire pair_1, pair_2, pair_3, pair_4;

    and AND_AB (ab, a, b);
    and AND_AC (ac, a, c);
    and AND_AD (ad, a, d);
    and AND_BC (bc, b, c);
    and AND_BD (bd, b, d);
    and AND_CD (cd, c, d);

    or OR_PAIR_1 (pair_1, ab, ac);
    or OR_PAIR_2 (pair_2, ad, bc);
    or OR_PAIR_3 (pair_3, bd, cd);
    or OR_PAIR_4 (pair_4, pair_1, pair_2);
    or OR_OUTPUT (y, pair_4, pair_3);
endmodule
