#include "labneura/eda/verilog_callable_execution.h"
#include "labneura/eda/verilog_syntax.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace {

using namespace labneura::eda;

void require(bool condition, std::string_view message) {
    if (!condition)
        throw std::runtime_error(std::string(message));
}

VerilogStatement statement(VerilogStatementKind kind, size_t offset = 1) {
    VerilogStatement result;
    result.kind = kind;
    result.location.byte_offset = offset;
    return result;
}

void test_nested_control_transfers_and_return() {
    VerilogStatement root = statement(VerilogStatementKind::Block);
    auto loop = std::make_unique<VerilogStatement>(statement(VerilogStatementKind::Loop));
    auto branch = std::make_unique<VerilogStatement>(statement(VerilogStatementKind::If));
    branch->first_statement = std::make_unique<VerilogStatement>(
        statement(VerilogStatementKind::Continue, 12));
    branch->second_statement = std::make_unique<VerilogStatement>(
        statement(VerilogStatementKind::Break, 13));
    loop->first_statement = std::move(branch);
    root.statements.push_back(std::move(loop));
    root.statements.push_back(std::make_unique<VerilogStatement>(
        statement(VerilogStatementKind::Return, 20)));

    const auto result = analyze_verilog_control_flow(root);
    require(result.valid && result.diagnostics.empty(),
            "loop-targeted break/continue were incorrectly rejected");
    require(result.maximum_loop_nesting == 1 && result.may_return,
            "nested loop depth or return presence was not recorded");
}

void test_invalid_control_transfer_locations() {
    VerilogStatement root = statement(VerilogStatementKind::Block);
    root.statements.push_back(std::make_unique<VerilogStatement>(
        statement(VerilogStatementKind::Break, 31)));
    root.statements.push_back(std::make_unique<VerilogStatement>(
        statement(VerilogStatementKind::Continue, 41)));

    const auto result = analyze_verilog_control_flow(root);
    require(!result.valid && result.diagnostics.size() == 2,
            "out-of-loop control transfers were not diagnosed");
    require(result.diagnostics[0].code == "BreakOutsideLoop" &&
                result.diagnostics[0].location.byte_offset == 31 &&
                result.diagnostics[1].code == "ContinueOutsideLoop" &&
                result.diagnostics[1].location.byte_offset == 41,
            "control-transfer diagnostics lost their source locations");
}

void test_nested_loop_depth_and_unsupported_classification() {
    VerilogStatement outer = statement(VerilogStatementKind::Loop);
    outer.first_statement = std::make_unique<VerilogStatement>(
        statement(VerilogStatementKind::Loop));
    outer.first_statement->first_statement = std::make_unique<VerilogStatement>(
        statement(VerilogStatementKind::Wait));
    const auto result = analyze_verilog_control_flow(outer);
    require(result.valid && result.maximum_loop_nesting == 2 &&
                result.contains_unsupported,
            "nested depth or unsupported statement classification is incorrect");
}

void test_timing_nodes_are_structurally_known() {
    VerilogStatement delay = statement(VerilogStatementKind::DelayControl);
    delay.first_statement = std::make_unique<VerilogStatement>(
        statement(VerilogStatementKind::Empty));
    const auto result = analyze_verilog_control_flow(delay);
    require(result.valid && !result.contains_unsupported,
            "timing-control AST was misclassified as an unknown statement");
}

} // namespace

int main() {
    try {
        test_nested_control_transfers_and_return();
        test_invalid_control_transfer_locations();
        test_nested_loop_depth_and_unsupported_classification();
        test_timing_nodes_are_structurally_known();
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
