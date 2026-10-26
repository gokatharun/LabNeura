module tick_generator #(
    parameter integer CLOCKS_PER_TICK = 50000000
) (
    input wire clk,
    input wire reset,
    output wire tick
);
    localparam integer COUNTER_WIDTH =
        (CLOCKS_PER_TICK <= 1) ? 1 : $clog2(CLOCKS_PER_TICK);

    reg [COUNTER_WIDTH-1:0] counter;

    assign tick = (CLOCKS_PER_TICK <= 1) ||
                  (counter == CLOCKS_PER_TICK - 1);

    always @(posedge clk or posedge reset) begin
        if (reset)
            counter <= 0;
        else if (CLOCKS_PER_TICK <= 1 || tick)
            counter <= 0;
        else
            counter <= counter + 1'b1;
    end
endmodule
