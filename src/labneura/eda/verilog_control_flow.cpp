#include "labneura/eda/verilog_callable_execution.h"

#include "labneura/eda/verilog_syntax.h"

namespace labneura::eda {
namespace {

class ControlFlowAnalyzer {
public:
    void visit(const VerilogStatement& statement, size_t loop_depth) {
        switch (statement.kind) {
            case VerilogStatementKind::Block:
                for (const auto& child : statement.statements)
                    if (child)
                        visit(*child, loop_depth);
                return;
            case VerilogStatementKind::If:
                visit_optional(statement.first_statement.get(), loop_depth);
                visit_optional(statement.second_statement.get(), loop_depth);
                return;
            case VerilogStatementKind::Case:
                for (const auto& item : statement.case_items)
                    visit_optional(item.statement.get(), loop_depth);
                return;
            case VerilogStatementKind::Loop: {
                const size_t nested_depth = loop_depth + 1;
                result.maximum_loop_nesting =
                    std::max(result.maximum_loop_nesting, nested_depth);
                visit_optional(statement.loop_initialization.get(), nested_depth);
                visit_optional(statement.first_statement.get(), nested_depth);
                visit_optional(statement.loop_step.get(), nested_depth);
                return;
            }
            case VerilogStatementKind::EventControl:
            case VerilogStatementKind::DelayControl:
                // Timing controls are recognized AST nodes. This analysis does
                // not imply that a timing context is executable by every caller.
                visit_optional(statement.first_statement.get(), loop_depth);
                return;
            case VerilogStatementKind::Return:
                result.may_return = true;
                return;
            case VerilogStatementKind::Break:
                if (loop_depth == 0)
                    error("BreakOutsideLoop",
                          "break statement is not nested inside a loop",
                          statement.location);
                return;
            case VerilogStatementKind::Continue:
                if (loop_depth == 0)
                    error("ContinueOutsideLoop",
                          "continue statement is not nested inside a loop",
                          statement.location);
                return;
            case VerilogStatementKind::Unsupported:
            case VerilogStatementKind::Wait:
                result.contains_unsupported = true;
                return;
            case VerilogStatementKind::ProceduralAssignment:
                if (statement.assignment_kind == VerilogAssignmentKind::Nonblocking)
                    result.contains_unsupported = true;
                return;
            case VerilogStatementKind::TaskCall:
            case VerilogStatementKind::Empty:
                return;
        }
        result.contains_unsupported = true;
    }

    VerilogControlFlowAnalysisResult result;

private:
    void visit_optional(const VerilogStatement* statement, size_t loop_depth) {
        if (statement)
            visit(*statement, loop_depth);
    }

    void error(std::string code, std::string message,
               const VerilogSyntaxLocation& location) {
        result.valid = false;
        result.diagnostics.push_back(
            {std::move(code), std::move(message), location});
    }
};

} // namespace

VerilogControlFlowAnalysisResult analyze_verilog_control_flow(
    const VerilogStatement& statement) {
    ControlFlowAnalyzer analyzer;
    analyzer.visit(statement, 0);
    return std::move(analyzer.result);
}

} // namespace labneura::eda
