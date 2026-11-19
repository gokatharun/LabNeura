#include "labneura/eda/verilog_process_analysis.h"

#include <iostream>
#include <stdexcept>
#include <string_view>

namespace {

using namespace labneura::eda;

void require(bool condition, std::string_view message) {
    if (!condition)
        throw std::runtime_error(std::string(message));
}

void test_source_order_classification_and_event_identity() {
    const auto parsed = parse_verilog_syntax(
        "module process_kinds(input logic clk, rst_n, a,\n"
        "                     output logic q);\n"
        "  logic n;\n"
        "  assign n = a;\n"
        "  always @(posedge clk or negedge rst_n)\n"
        "    if (!rst_n) q <= 1'b0; else q <= a;\n"
        "  always @(a) q = a;\n"
        "  always_comb if (a) q = a;\n"
        "  always_latch if (a) q = a;\n"
        "  initial q = 1'b0;\n"
        "endmodule\n");
    require(!parsed.has_errors(), "valid native process fixture failed to parse");
    const auto semantic = analyze_verilog_semantics(parsed.syntax);
    require(!semantic.has_errors(), "process fixture has semantic errors");
    const auto& module = parsed.syntax.modules.front();
    const auto result = discover_verilog_processes(module, semantic);

    require(result.diagnostics.empty() && result.processes.size() == 6,
            "process discovery lost an assignment or procedural block");
    require(result.processes[0].kind ==
                    VerilogProcessKind::ContinuousAssignment &&
                result.processes[0].classification ==
                    VerilogProcessClassification::ContinuousDriver &&
                result.processes[0].assignments.size() == 1 &&
                result.processes[0].assignments[0].kind ==
                    VerilogProcessAssignmentKind::Continuous,
            "continuous assignment process record is incorrect");

    const auto& edge_always = result.processes[1];
    require(edge_always.kind == VerilogProcessKind::Always &&
                edge_always.classification ==
                    VerilogProcessClassification::GenericAlways &&
                edge_always.has_explicit_event_control &&
                edge_always.events.size() == 2 &&
                edge_always.events[0].edge == VerilogEventEdge::Posedge &&
                edge_always.events[1].edge == VerilogEventEdge::Negedge &&
                edge_always.events[0].symbol && edge_always.events[1].symbol &&
                edge_always.events[0].symbol->name == "clk" &&
                edge_always.events[1].symbol->name == "rst_n",
            "generic always intent or event symbols were incorrectly classified");
    require(edge_always.assignments.size() == 2 &&
                edge_always.assignments[0].kind ==
                    VerilogProcessAssignmentKind::Nonblocking &&
                edge_always.assignments[1].kind ==
                    VerilogProcessAssignmentKind::Nonblocking,
            "nonblocking assignment categories were not retained");

    const auto& level_always = result.processes[2];
    require(level_always.classification ==
                    VerilogProcessClassification::GenericAlways &&
                level_always.events.size() == 1 &&
                level_always.events.front().edge == VerilogEventEdge::Any &&
                level_always.events.front().symbol &&
                level_always.events.front().symbol->name == "a" &&
                level_always.assignments.front().kind ==
                    VerilogProcessAssignmentKind::Blocking,
            "level-sensitive generic always classification is incorrect");
    require(result.processes[3].kind == VerilogProcessKind::AlwaysComb &&
                result.processes[3].classification ==
                    VerilogProcessClassification::CombinationalCandidate &&
                result.processes[3].implicit_sensitivity &&
                result.processes[4].kind == VerilogProcessKind::AlwaysLatch &&
                result.processes[4].classification ==
                    VerilogProcessClassification::LatchCandidate &&
                result.processes[4].implicit_sensitivity,
            "always_comb/always_latch intent was not preserved");
    require(result.processes[5].kind == VerilogProcessKind::Initial &&
                result.processes[5].classification ==
                    VerilogProcessClassification::InitializationProcess &&
                !result.processes[5].contains_unsupported_statement,
            "initial process was rejected instead of separately classified");

    for (size_t index = 1; index < result.processes.size(); ++index)
        require(result.processes[index - 1].location.byte_offset <
                    result.processes[index].location.byte_offset,
                "discovered processes are not in source order");
}

void test_wildcard_and_nested_timing_flags() {
    const auto parsed = parse_verilog_syntax(
        "module process_flags(input logic a, output logic q);\n"
        "  always @* q = a;\n"
        "  initial begin wait (a); #1 q = a; end\n"
        "endmodule\n");
    require(!parsed.has_errors(), "valid process flag fixture failed to parse");
    const auto semantic = analyze_verilog_semantics(parsed.syntax);
    const auto result =
        discover_verilog_processes(parsed.syntax.modules.front(), semantic);
    require(result.processes.size() == 2,
            "process flag discovery returned an incorrect process count");
    require(result.processes[0].has_explicit_event_control &&
                result.processes[0].wildcard_sensitivity &&
                result.processes[0].implicit_sensitivity,
            "wildcard sensitivity metadata was not retained");
    require(result.processes[1].kind == VerilogProcessKind::Initial &&
                result.processes[1].contains_timing_control &&
                result.processes[1].contains_unsupported_statement,
            "nested timing or unsupported statements were not recorded");
}

} // namespace

int main() {
    try {
        test_source_order_classification_and_event_identity();
        test_wildcard_and_nested_timing_flags();
    } catch (const std::exception& exception) {
        std::cerr << "EDA Verilog process analysis test failed: "
                  << exception.what() << '\n';
        return 1;
    }
    return 0;
}
