#include "labneura/eda/verilog_process_analysis.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace {
using namespace labneura::eda;
void require(bool condition, std::string_view message) {
    if (!condition) throw std::runtime_error(std::string(message));
}
void test_symbol_identity_reads_writes_and_control_expressions() {
    const auto parsed = parse_verilog_syntax(
        "module deps(input logic [3:0] a, b, input logic sel, output logic [3:0] q);\n"
        "  always_comb begin\n"
        "    if (sel) q = a[2:1]; else q = b;\n"
        "  end\n"
        "endmodule\n");
    require(!parsed.has_errors(), "dependency fixture did not parse");
    const auto semantic = analyze_verilog_semantics(parsed.syntax);
    const auto types = analyze_verilog_expression_types(parsed.syntax, semantic);
    const auto processes = discover_verilog_processes(parsed.syntax.modules.front(), semantic);
    require(processes.processes.size() == 1, "expected one process");
    const auto deps = analyze_verilog_process_dependencies(
        processes.processes.front(), parsed.syntax, semantic, types);
    auto contains = [](const auto& symbols, std::string_view name) {
        for (const auto* symbol : symbols) if (symbol && symbol->name == name) return true;
        return false;
    };
    require(contains(deps.raw_reads, "a") && contains(deps.raw_reads, "b") &&
            contains(deps.raw_reads, "sel"), "RHS/condition reads were lost");
    require(contains(deps.writes, "q"), "assignment destination write was lost");
    require(contains(deps.implicit_sensitivity, "sel") &&
            contains(deps.implicit_sensitivity, "a"), "implicit sensitivity differs from raw reads");
    bool range_seen = false;
    for (const auto& access : deps.accesses)
        if (access.symbol && access.symbol->name == "a" && access.read &&
            access.lsb == 1 && access.width == 2) range_seen = true;
    require(range_seen, "constant selected read range was not retained");
}
void test_dynamic_select_and_explicit_event_are_separate() {
    const auto parsed = parse_verilog_syntax(
        "module deps(input logic [7:0] data, input logic idx, output logic q);\n"
        "  always @(idx) q = data[idx];\n"
        "endmodule\n");
    require(!parsed.has_errors(), "dynamic dependency fixture did not parse");
    const auto semantic = analyze_verilog_semantics(parsed.syntax);
    const auto types = analyze_verilog_expression_types(parsed.syntax, semantic);
    const auto process = discover_verilog_processes(parsed.syntax.modules.front(), semantic).processes.front();
    const auto deps = analyze_verilog_process_dependencies(process, parsed.syntax, semantic, types);
    require(deps.implicit_sensitivity.empty(), "explicit event list became implicit sensitivity");
    bool dynamic = false;
    for (const auto& access : deps.accesses)
        if (access.symbol && access.symbol->name == "data" && access.read && access.dynamic_select)
            dynamic = true;
    require(dynamic, "dynamic bit select was not represented conservatively");
}

void test_function_body_captured_dependencies() {
    const auto parsed = parse_verilog_syntax(
        "module deps(input logic a, output logic q);\n"
        "  function logic f(input logic value); f = value & a; endfunction\n"
        "  always_comb q = f(1'b1);\n"
        "endmodule\n");
    require(!parsed.has_errors(), "callable dependency fixture did not parse");
    const auto semantic = analyze_verilog_semantics(parsed.syntax);
    require(!semantic.has_errors(), "callable dependency fixture has semantic errors");
    const auto types = analyze_verilog_expression_types(parsed.syntax, semantic);
    const auto process =
        discover_verilog_processes(parsed.syntax.modules.front(), semantic)
            .processes.front();
    const auto deps = analyze_verilog_process_dependencies(
        process, parsed.syntax, semantic, types);
    require(deps.knowledge == VerilogAnalysisKnowledge::Known &&
                !deps.unresolved_callable_side_effects,
            "known same-module function was not summarized");
    require(std::any_of(deps.raw_reads.begin(), deps.raw_reads.end(),
                        [](const auto* symbol) {
                            return symbol && symbol->name == "a";
                        }),
            "function captured signal read was omitted from dependencies");
}
}
int main() {
    try {
        test_symbol_identity_reads_writes_and_control_expressions();
        test_dynamic_select_and_explicit_event_are_separate();
        test_function_body_captured_dependencies();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
