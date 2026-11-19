#include "labneura/eda/verilog_process_analysis.h"

#include <stdexcept>
#include <string_view>

namespace {
using namespace labneura::eda;
void require(bool ok, std::string_view why) { if (!ok) throw std::runtime_error(std::string(why)); }

VerilogSequentialProcessAnalysis analyze(const std::string& source, size_t index = 0) {
    auto parsed = parse_verilog_syntax(source);
    require(!parsed.has_errors(), "fixture did not parse");
    auto sem = analyze_verilog_semantics(parsed.syntax);
    require(!sem.has_errors(), "fixture has semantic errors");
    auto types = analyze_verilog_expression_types(parsed.syntax, sem);
    auto discovered = discover_verilog_processes(parsed.syntax.modules.front(), sem);
    require(index < discovered.processes.size(), "fixture process was not discovered");
    auto deps = analyze_verilog_process_dependencies(discovered.processes[index], parsed.syntax, sem, types);
    return analyze_verilog_sequential_process(discovered.processes[index], sem, types, deps);
}

void test_async_reset_requires_matching_condition() {
    auto a = analyze("module m(input logic clk, rst_n, d, output logic q);\n"
                     "always @(posedge clk or negedge rst_n)\n"
                     " if (!rst_n) q <= 0; else q <= d;\nendmodule\n");
    require(a.clocks.size() == 1 && a.clocks[0].edge == VerilogEventEdge::Posedge,
            "posedge clock was not identified");
    require(a.resets.size() == 1 && a.resets[0].asynchronous && !a.resets[0].active_high,
            "matching active-low asynchronous reset was not identified");
    require(a.has_nonblocking_assignments && !a.has_blocking_assignments,
            "assignment categories are wrong");
}

void test_active_low_else_branch_async_reset() {
    auto a = analyze("module m(input logic clk, rst_n, d, output logic q);\n"
                     "always @(posedge clk or negedge rst_n)\n"
                     " if (rst_n) q <= d; else q <= 1'b0;\nendmodule\n");
    require(a.clocks.size() == 1 && a.resets.size() == 1 &&
                a.resets[0].asynchronous && !a.resets[0].active_high &&
                a.resets[0].reset_value_is_constant,
            "active-low reset in the else branch was not recognized structurally");
}

void test_extra_edge_without_reset_test_is_ambiguous() {
    auto a = analyze("module m(input logic clk, aux, d, output logic q);\n"
                     "always @(posedge clk or posedge aux) q <= d;\nendmodule\n");
    require(a.has_ambiguous_clock_or_reset && a.resets.empty(),
            "unmatched extra edge was incorrectly treated as an asynchronous reset");

    auto nonconstant_reset = analyze(
        "module m(input logic clk, rst_n, d, output logic q);\n"
        "always @(posedge clk or negedge rst_n)\n"
        " if (!rst_n) q <= d; else q <= d;\nendmodule\n");
    require(nonconstant_reset.has_ambiguous_clock_or_reset &&
                nonconstant_reset.resets.empty(),
            "nonconstant reset branch was forced into an asynchronous reset candidate");
}

void test_blocking_and_simple_clock() {
    auto a = analyze("module m(input logic clk, d, output logic q);\n"
                     "always @(negedge clk) q = d;\nendmodule\n");
    require(a.clocks.size() == 1 && a.clocks[0].edge == VerilogEventEdge::Negedge,
            "negedge clock was not identified");
    require(a.has_blocking_assignments && !a.has_nonblocking_assignments,
            "blocking assignment category is wrong");
}

void test_synchronous_reset_candidate() {
    auto a = analyze("module m(input logic clk, rst, d, output logic q);\n"
                     "always @(posedge clk)\n"
                     " if (rst) q <= 0; else q <= d;\nendmodule\n");
    require(a.clocks.size() == 1 && a.resets.size() == 1 &&
            a.resets[0].synchronous && !a.resets[0].asynchronous &&
            a.resets[0].active_high,
            "synchronous active-high reset candidate was not identified");
}
}

int main() {
    test_async_reset_requires_matching_condition();
    test_active_low_else_branch_async_reset();
    test_extra_edge_without_reset_test_is_ambiguous();
    test_blocking_and_simple_clock();
    test_synchronous_reset_candidate();
    return 0;
}
