#pragma once

#include "labneura/eda/verilog_semantic.h"
#include "labneura/eda/verilog_type.h"

#include <optional>
#include <string>
#include <vector>

namespace labneura::eda {

struct VerilogCallableArgumentBinding {
    size_t formal_index = 0;
    const VerilogSemanticSymbol* formal = nullptr;
    const VerilogExpression* actual = nullptr;
};

struct VerilogCallableBindingDiagnostic {
    VerilogTypeDiagnosticSeverity severity = VerilogTypeDiagnosticSeverity::Error;
    std::string code;
    std::string message;
    VerilogSyntaxLocation location;
    std::optional<VerilogSyntaxLocation> related_location;
};

struct VerilogCallableBindingResult {
    std::vector<VerilogCallableArgumentBinding> arguments;
    std::vector<VerilogCallableBindingDiagnostic> diagnostics;
    bool is_valid = true;
};

VerilogCallableBindingResult bind_verilog_callable_arguments(
    const VerilogSemanticSymbol& callable, const VerilogExpression& call,
    const VerilogTypeAnalysisResult& type_analysis);

const VerilogCallableDeclaration* find_verilog_callable_declaration(
    const VerilogCompilationUnit& syntax, const VerilogSemanticSymbol& callable);

} // namespace labneura::eda
