module phase_timer (
    input wire clk,
    input wire reset,
    input wire tick,
    input wire [31:0] duration,
    output wire done
);
    reg [31:0] elapsed_ticks;

    assign done = (duration == 0) || (elapsed_ticks >= duration - 1'b1);

    always @(posedge clk or posedge reset) begin
        if (reset)
            elapsed_ticks <= 0;
        else if (tick) begin
            if (done)
                elapsed_ticks <= 0;
            else
                elapsed_ticks <= elapsed_ticks + 1'b1;
        end
    end
endmodule
