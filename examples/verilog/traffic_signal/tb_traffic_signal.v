module tb_traffic_signal;
    reg clk;
    reg reset;
    wire [2:0] main_lights;
    wire [2:0] side_lights;

    localparam [2:0] RED = 3'b100;
    localparam [2:0] YELLOW = 3'b010;
    localparam [2:0] GREEN = 3'b001;

    traffic_signal #(
        .CLOCKS_PER_TICK(1),
        .GREEN_TICKS(3),
        .YELLOW_TICKS(2),
        .ALL_RED_TICKS(1)
    ) dut (
        .clk(clk),
        .reset(reset),
        .main_lights(main_lights),
        .side_lights(side_lights)
    );

    always #5 clk = ~clk;

    task check_lights;
        input [2:0] expected_main;
        input [2:0] expected_side;
        begin
            if (main_lights !== expected_main ||
                side_lights !== expected_side)
                $fatal(1, "Expected main=%b side=%b, got main=%b side=%b",
                       expected_main, expected_side, main_lights, side_lights);
            if (main_lights == GREEN && side_lights == GREEN)
                $fatal(1, "Both directions must never be green");
        end
    endtask

    task advance_ticks;
        input integer count;
        integer tick_index;
        begin
            for (tick_index = 0; tick_index < count; tick_index = tick_index + 1)
                @(posedge clk);
            #1;
        end
    endtask

    initial begin
        clk = 1'b0;
        reset = 1'b1;

        repeat (2) @(posedge clk);
        #1;
        check_lights(GREEN, RED);

        @(negedge clk);
        reset = 1'b0;

        advance_ticks(2);
        check_lights(GREEN, RED);
        advance_ticks(1);
        check_lights(YELLOW, RED);
        advance_ticks(2);
        check_lights(RED, RED);
        advance_ticks(1);
        check_lights(RED, GREEN);
        advance_ticks(3);
        check_lights(RED, YELLOW);
        advance_ticks(2);
        check_lights(RED, RED);
        advance_ticks(1);
        check_lights(GREEN, RED);

        $display("PASS: traffic signal phase and safety checks");
        $finish;
    end

    initial begin
        #1000;
        $fatal(1, "FAIL: traffic signal test timed out");
    end
endmodule
