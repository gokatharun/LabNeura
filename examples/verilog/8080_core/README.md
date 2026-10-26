# Intel 8080 CPU Core

This folder contains a complete documented-instruction-set Intel 8080 CPU
implementation, its ALU, and a small memory-backed simulation testbench. It is
a CPU-only example: RAM and I/O are external to the processor core.

The CPU and ALU are copied from
[YosysHQ/prjpeppercorn-test-cases](https://github.com/YosysHQ/prjpeppercorn-test-cases/tree/539ccbf035a23cf474ee1310e05222efbea05e8d/075-invaders)
at commit `539ccbf035a23cf474ee1310e05222efbea05e8d`. They are provided under
the included MIT license. The unrelated Space Invaders game ROM and board
peripherals are not included.

The test program exercises register-pair loads, memory reads and writes,
arithmetic and flags, a conditional jump, stack operations, a subroutine call
and return, and `HLT`. Run it from this directory with Icarus Verilog:

```sh
iverilog -g2012 -o sim i8080.v alu.v tb_i8080.v
vvp sim
```

The core uses a ready/valid-style external bus (`i_ready`, `o_read`,
`o_write`, `o_addr`, and data signals). The testbench models synchronous
memory that responds to each read or write request. The instruction decoder
covers the documented 8080 families: register and memory transfers; 8-bit
arithmetic, logic, and compare; 16-bit pair operations; rotates and decimal
adjust; conditional and unconditional jumps, calls, and returns; stack
operations; restart; I/O; interrupt enable/disable; and halt/no-op.

The testbench is a smoke test for representative instruction paths, not an
exhaustive 8080 conformance suite. This example does not claim silicon-cycle-
level timing validation.
