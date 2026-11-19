#pragma once

#include "labneura/eda/verilog_constant.h"
#include "labneura/eda/verilog_semantic.h"
#include "labneura/eda/verilog_type.h"

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace labneura::eda {

struct VerilogElaborationOptions {
    std::optional<std::string> top_module;
    VerilogConstantEvaluationOptions constant_options;
    size_t maximum_generate_iterations = 1024;
};

struct VerilogElaboratedGenerateScope;

struct VerilogElaboratedInstance {
    std::string instance_name;
    std::string hierarchy_path;
    VerilogSyntaxLocation location;
    const VerilogModuleDeclaration* module_definition = nullptr;
    const VerilogSemanticSymbol* module_symbol = nullptr;
    VerilogElaboratedInstance* parent = nullptr;
    VerilogElaboratedGenerateScope* generated_scope_parent = nullptr;
    VerilogConstantEnvironment parameters;
    VerilogTypeEnvironment parameter_types;
    std::vector<const VerilogVariableDeclaration*> generated_declarations;
    std::vector<const VerilogContinuousAssignment*> generated_assignments;
    std::vector<std::unique_ptr<VerilogElaboratedInstance>> children;
    std::vector<std::unique_ptr<VerilogElaboratedGenerateScope>> generate_scopes;
};

struct VerilogElaboratedGenerateScope {
    std::string name;
    std::string hierarchy_path;
    VerilogSyntaxLocation location;
    VerilogConstantEnvironment parameters;
    VerilogTypeEnvironment parameter_types;
    std::vector<const VerilogVariableDeclaration*> declarations;
    std::vector<const VerilogContinuousAssignment*> assignments;
    VerilogElaboratedInstance* parent_instance = nullptr;
    VerilogElaboratedGenerateScope* parent_scope = nullptr;
    std::vector<std::unique_ptr<VerilogElaboratedInstance>> children;
    std::vector<std::unique_ptr<VerilogElaboratedGenerateScope>> generate_scopes;
};

struct VerilogElaborationDiagnostic {
    std::string code;
    std::string message;
    VerilogSyntaxLocation location;
    std::optional<VerilogSyntaxLocation> related_location;
};

struct VerilogElaborationResult {
    VerilogSemanticAnalysisResult semantic;
    std::unique_ptr<VerilogElaboratedInstance> top;
    std::vector<std::string> top_module_candidates;
    std::vector<VerilogElaborationDiagnostic> diagnostics;

    bool has_errors() const noexcept { return !diagnostics.empty(); }
};

VerilogElaborationResult elaborate_verilog_design(
    const VerilogCompilationUnit& syntax,
    VerilogElaborationOptions options = {});

} // namespace labneura::eda
