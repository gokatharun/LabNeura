module m(
    input logic a,
    input logic b,
    input logic cin,
    output logic and_result,
    output logic or_result,
    output logic xor_result,
    output logic half_sum,
    output logic half_carry,
    output logic full_sum,
    output logic full_carry
);
    assign and_result = a & b;
    assign or_result = a | b;
    assign xor_result = a ^ b;
    assign half_sum = a ^ b;
    assign half_carry = a & b;
    assign full_sum = a ^ b ^ cin;
    assign full_carry = (a & b) | (a & cin) | (b & cin);
endmodule
