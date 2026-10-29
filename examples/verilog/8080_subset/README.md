# Educational 8080-Style CPU Subset

This is a small, synthesizable teaching example. It is **not** a complete
Intel 8080 implementation. For a separate full documented-instruction-set CPU
core example, see [`../8080_core`](../8080_core/README.md).

The CPU has an 8-bit accumulator and B register, a 16-bit program counter, and
an external read-only program-memory interface. It supports:

| Opcode | Instruction | Behavior |
|---|---|---|
| `00` | `NOP` | Do nothing |
| `06` | `MVI B, d8` | Load the next byte into B |
| `3E` | `MVI A, d8` | Load the next byte into A |
| `78` | `MOV A, B` | Copy B into A |
| `80` | `ADD B` | Add B to A |
| `76` | `HLT` | Stop execution |

The test program loads 2 into A, loads 3 into B, adds B to A, then halts. The
testbench checks that the final values are A = 5 and B = 3.

Run the test from this directory with Icarus Verilog:

```sh
iverilog -g2012 -o sim cpu8080_subset.v tb_cpu8080_subset.v
vvp sim
```

The behavioral RTL can be read by LabNeura's project reader with
`parse_verilog` with the project directory.
