# Modular Traffic Signal Controller

This small, synthesizable example controls two intersecting traffic directions:
the main road and the side road. The design is split into four reusable
modules:

- `tick_generator` divides the input clock into slower control ticks.
- `phase_timer` measures how long the current phase lasts.
- `traffic_fsm` sequences the lights.
- `traffic_signal` connects the divider and controller.

Each light output is three bits in `{red, yellow, green}` order. The sequence
is main green, main yellow, all red, side green, side yellow, all red, then
repeat. The all-red phases prevent both roads from being given right-of-way
during a transition.

Defaults assume a 50 MHz input clock and a one-second control tick: 10 seconds
green, 3 seconds yellow, and 1 second all-red. Override the parameters for
another clock or timing. The testbench uses short durations so the entire
sequence can be simulated quickly.

Run with Icarus Verilog from this directory:

```sh
iverilog -g2012 -o sim tick_generator.v phase_timer.v traffic_fsm.v traffic_signal.v tb_traffic_signal.v
vvp sim
```

LabNeura can also parse the project by passing its directory to `parse_verilog`:

```python
import labneura.eda as eda

project = eda.parse_verilog("examples/verilog/traffic_signal")
print(project.module_count)
```
