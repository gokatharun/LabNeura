module traffic_signal #(
    parameter integer CLOCKS_PER_TICK = 50000000,
    parameter integer GREEN_TICKS = 10,
    parameter integer YELLOW_TICKS = 3,
    parameter integer ALL_RED_TICKS = 1
) (
    input wire clk,
    input wire reset,
    output wire [2:0] main_lights,
    output wire [2:0] side_lights
);
    wire tick;

    tick_generator #(
        .CLOCKS_PER_TICK(CLOCKS_PER_TICK)
    ) clock_divider (
        .clk(clk),
        .reset(reset),
        .tick(tick)
    );

    traffic_fsm #(
        .GREEN_TICKS(GREEN_TICKS),
        .YELLOW_TICKS(YELLOW_TICKS),
        .ALL_RED_TICKS(ALL_RED_TICKS)
    ) controller (
        .clk(clk),
        .reset(reset),
        .tick(tick),
        .main_lights(main_lights),
        .side_lights(side_lights)
    );
endmodule
