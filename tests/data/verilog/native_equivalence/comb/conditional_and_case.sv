module m(
    input logic [1:0] a,
    input logic [1:0] b,
    input logic sel,
    input logic [1:0] choice,
    output logic [1:0] q,
    output logic [1:0] selected
);
    logic [1:0] temporary;
    always_comb begin
        temporary = a;
        temporary = temporary & b;
        if (sel)
            q = temporary;
        else
            q = a;

        case (choice)
            2'b00: selected = a;
            2'b01: selected = b;
            2'b10: selected = a ^ b;
            default: selected = a | b;
        endcase
    end
endmodule
