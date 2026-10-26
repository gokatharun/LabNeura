// Assert the alarm for smoke-and-heat together, or for a manual trigger.
module alarm_logic (smoke, heat, manual, alarm);
    input wire smoke, heat, manual;
    output wire alarm;
    wire smoke_and_heat;

    and AND1 (smoke_and_heat, smoke, heat);
    or OR1 (alarm, smoke_and_heat, manual);
endmodule
