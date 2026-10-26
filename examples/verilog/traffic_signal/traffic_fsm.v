module traffic_fsm #(
    parameter integer GREEN_TICKS = 10,
    parameter integer YELLOW_TICKS = 3,
    parameter integer ALL_RED_TICKS = 1
) (
    input wire clk,
    input wire reset,
    input wire tick,
    output reg [2:0] main_lights,
    output reg [2:0] side_lights
);
    localparam [2:0] MAIN_GREEN = 3'd0;
    localparam [2:0] MAIN_YELLOW = 3'd1;
    localparam [2:0] ALL_RED_TO_SIDE = 3'd2;
    localparam [2:0] SIDE_GREEN = 3'd3;
    localparam [2:0] SIDE_YELLOW = 3'd4;
    localparam [2:0] ALL_RED_TO_MAIN = 3'd5;

    localparam [2:0] RED = 3'b100;
    localparam [2:0] YELLOW = 3'b010;
    localparam [2:0] GREEN = 3'b001;

    reg [2:0] state;
    reg [2:0] next_state;
    wire [31:0] phase_duration;
    wire phase_done;

    assign phase_duration =
        (state == MAIN_GREEN || state == SIDE_GREEN) ? GREEN_TICKS :
        (state == MAIN_YELLOW || state == SIDE_YELLOW) ? YELLOW_TICKS :
        ALL_RED_TICKS;

    phase_timer timer (
        .clk(clk),
        .reset(reset),
        .tick(tick),
        .duration(phase_duration),
        .done(phase_done)
    );

    always @* begin
        case (state)
            MAIN_GREEN: next_state = MAIN_YELLOW;
            MAIN_YELLOW: next_state = ALL_RED_TO_SIDE;
            ALL_RED_TO_SIDE: next_state = SIDE_GREEN;
            SIDE_GREEN: next_state = SIDE_YELLOW;
            SIDE_YELLOW: next_state = ALL_RED_TO_MAIN;
            ALL_RED_TO_MAIN: next_state = MAIN_GREEN;
            default: next_state = MAIN_GREEN;
        endcase
    end

    always @(posedge clk or posedge reset) begin
        if (reset)
            state <= MAIN_GREEN;
        else if (tick && phase_done)
            state <= next_state;
    end

    always @* begin
        main_lights = RED;
        side_lights = RED;

        case (state)
            MAIN_GREEN: main_lights = GREEN;
            MAIN_YELLOW: main_lights = YELLOW;
            SIDE_GREEN: side_lights = GREEN;
            SIDE_YELLOW: side_lights = YELLOW;
            default: begin
                main_lights = RED;
                side_lights = RED;
            end
        endcase
    end
endmodule
