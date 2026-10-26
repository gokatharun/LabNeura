// Basic two-input OR gate.
module or_gate (a, b, y);
    input wire a, b;
    output wire y;

    or OR1 (y, a, b);
endmodule
