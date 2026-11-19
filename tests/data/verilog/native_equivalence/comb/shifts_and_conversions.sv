module m(
    input logic [2:0] value,
    input logic signed [2:0] signed_value,
    input logic [2:0] unsigned_value,
    output logic [3:0] left_shifted,
    output logic signed [4:0] sign_extended,
    output logic [4:0] zero_extended
);
    assign left_shifted = value << 1;
    assign sign_extended = signed_value;
    assign zero_extended = unsigned_value;
endmodule
