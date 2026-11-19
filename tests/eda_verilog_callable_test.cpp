#include "labneura/eda/verilog_semantic.h"
#include "labneura/eda/verilog_type.h"

#include <slang/syntax/SyntaxTree.h>

#include <algorithm>
#include <cctype>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

using namespace labneura::eda;

void require(bool condition, std::string_view message) {
    if (!condition)
        throw std::runtime_error(std::string(message));
}

bool contains_case_insensitive(std::string_view text, std::string_view needle) {
    return std::search(text.begin(), text.end(), needle.begin(), needle.end(),
                       [](unsigned char lhs, unsigned char rhs) {
                           return std::tolower(lhs) == std::tolower(rhs);
                       }) != text.end();
}

const VerilogCallableDeclaration& find_callable(
    const std::vector<VerilogCallableDeclaration>& callables, std::string_view name) {
    const auto found = std::find_if(callables.begin(), callables.end(),
                                    [&](const auto& callable) {
                                        return callable.name == name;
                                    });
    if (found == callables.end())
        throw std::runtime_error("expected callable declaration was not parsed");
    return *found;
}

const VerilogExpression* find_call(const VerilogExpression* expression,
                                   std::string_view callee) {
    if (!expression)
        return nullptr;
    if (expression->kind == VerilogExpressionKind::Call && expression->first &&
        expression->first->text == callee)
        return expression;
    if (const auto* found = find_call(expression->first.get(), callee))
        return found;
    if (const auto* found = find_call(expression->second.get(), callee))
        return found;
    if (const auto* found = find_call(expression->third.get(), callee))
        return found;
    for (const auto& operand : expression->operands)
        if (const auto* found = find_call(operand.get(), callee))
            return found;
    return nullptr;
}

const VerilogStatement* find_task_call(const VerilogStatement* statement,
                                       std::string_view callee) {
    if (!statement)
        return nullptr;
    if (statement->kind == VerilogStatementKind::TaskCall &&
        statement->call_expression &&
        statement->call_expression->kind == VerilogExpressionKind::Call &&
        statement->call_expression->first &&
        statement->call_expression->first->text == callee)
        return statement;
    if (const auto* found = find_task_call(statement->first_statement.get(), callee))
        return found;
    if (const auto* found = find_task_call(statement->second_statement.get(), callee))
        return found;
    for (const auto& nested : statement->statements)
        if (const auto* found = find_task_call(nested.get(), callee))
            return found;
    for (const auto& item : statement->case_items)
        if (const auto* found = find_task_call(item.statement.get(), callee))
            return found;
    return nullptr;
}

bool mentions(const VerilogSemanticDiagnostic& diagnostic,
              std::string_view keyword) {
    return contains_case_insensitive(diagnostic.code, keyword) ||
           contains_case_insensitive(diagnostic.message, keyword);
}

bool has_callable_diagnostic(const VerilogSemanticAnalysisResult& result,
                             std::string_view first, std::string_view second = {}) {
    return std::any_of(result.diagnostics.begin(), result.diagnostics.end(),
                       [&](const auto& diagnostic) {
                           return mentions(diagnostic, first) &&
                                  (second.empty() || mentions(diagnostic, second));
                       });
}

void test_declarations_bodies_and_locations() {
    const auto parsed = parse_verilog_syntax(
        "module callable_probe(input logic [7:0] x, output logic [7:0] y);\n"
        "  function automatic logic [7:0] widen(input logic [3:0] value);\n"
        "    logic [7:0] temporary;\n"
        "    begin : body\n"
        "      if (value[0]) begin\n"
        "        temporary = {4'b0, value};\n"
        "      end else begin\n"
        "        temporary = 8'h00;\n"
        "      end\n"
        "      widen = temporary;\n"
        "    end\n"
        "  endfunction\n"
        "  task automatic transfer(input logic [7:0] data,\n"
        "                         output logic done, ref logic [3:0] scratch [1:0]);\n"
        "    begin\n"
        "      scratch = data[3:0];\n"
        "      done = 1'b1;\n"
        "    end\n"
        "  endtask\n"
        "  logic done;\n"
        "  logic [3:0] scratch [1:0];\n"
        "  assign y = widen(x[3:0]);\n"
        "  always_comb begin\n"
        "    transfer(x, done, scratch);\n"
        "  end\n"
        "endmodule\n"
        "interface callable_if;\n"
        "  function logic [3:0] nibble(input logic [7:0] value);\n"
        "    nibble = value[3:0];\n"
        "  endfunction\n"
        "endinterface\n"
        "package callable_pkg;\n"
        "  function automatic logic [7:0] identity(input logic [7:0] value);\n"
        "    identity = value;\n"
        "  endfunction\n"
        "endpackage\n");
    require(!parsed.has_errors(), "callable declaration fixture should parse cleanly");
    require(parsed.syntax.modules.size() == 1 && parsed.syntax.interfaces.size() == 1 &&
                parsed.syntax.packages.size() == 1,
            "callables were not retained on module, interface, and package declarations");

    const auto& module = parsed.syntax.modules.front();
    const auto& widen = find_callable(module.callables, "widen");
    const auto& transfer = find_callable(module.callables, "transfer");
    require(widen.kind == VerilogCallableKind::Function &&
                widen.return_type == "logic" &&
                widen.return_packed_dimensions.size() == 1 &&
                widen.formal_arguments.size() == 1 &&
                widen.formal_arguments[0].name == "value" &&
                widen.formal_arguments[0].direction == VerilogPortDirection::Input &&
                widen.formal_arguments[0].type == "logic" &&
                widen.formal_arguments[0].packed_dimensions.size() == 1,
            "function kind, return type/range, or typed input formal is incorrect");
    require(transfer.kind == VerilogCallableKind::Task &&
                transfer.formal_arguments.size() == 3 &&
                transfer.formal_arguments[0].direction == VerilogPortDirection::Input &&
                transfer.formal_arguments[1].direction == VerilogPortDirection::Output &&
                transfer.formal_arguments[2].direction == VerilogPortDirection::Ref &&
                transfer.formal_arguments[2].name == "scratch" &&
                transfer.formal_arguments[2].packed_dimensions.size() == 1 &&
                transfer.formal_arguments[2].unpacked_dimensions.size() == 1,
            "task formal order, types, or directions are incorrect");
    require(widen.local_declarations.size() == 1 &&
                widen.local_declarations.front().name == "temporary" &&
                widen.body && widen.body->kind == VerilogStatementKind::Block &&
                widen.body->statements.size() == 1 &&
                widen.body->statements.front()->kind == VerilogStatementKind::Block &&
                widen.body->statements.front()->statements.size() == 2 &&
                widen.body->statements.front()->statements.front()->kind ==
                    VerilogStatementKind::If &&
                widen.location.source && widen.location.location.line == 2 &&
                transfer.location.source && transfer.location.location.line == 13,
            "callable local declaration, nested body, or declaration location was lost");
    require(find_callable(parsed.syntax.interfaces.front().callables, "nibble").kind ==
                    VerilogCallableKind::Function &&
                find_callable(parsed.syntax.packages.front().callables, "identity").kind ==
                    VerilogCallableKind::Function,
            "interface or package callable declaration was not parsed");

    const auto* task_call = find_task_call(module.procedural_blocks.front().body.get(),
                                           "transfer");
    require(task_call, "task invocation was not parsed as a TaskCall statement");
    require(task_call->call_expression &&
                task_call->call_expression->operands.size() == 3 &&
                task_call->call_expression->first->kind ==
                    VerilogExpressionKind::Identifier,
            "task invocation is not represented as a TaskCall with a Call expression");
}

void test_nonassignable_output_inout_ref_actual_diagnostics() {
    const std::string source =
        "module invalid_output_actual;\n"
        "  task automatic capture(output logic out_value,\n"
        "                        inout logic io_value, ref logic ref_value);\n"
        "    begin\n"
        "      out_value = 1'b0;\n"
        "      io_value = 1'b0;\n"
        "      ref_value = 1'b0;\n"
        "    end\n"
        "  endtask\n"
        "  initial begin\n"
        "    capture(1'b1, 1'b0 + 1'b1, 1'b0);\n"
        "  end\n"
        "endmodule\n";
    const auto parsed = parse_verilog_syntax(source);
    require(!parsed.has_errors(),
            "non-assignable output/inout/ref-argument fixture should parse cleanly");
    const auto semantic = analyze_verilog_semantics(parsed.syntax);
    const auto count = std::count_if(
        semantic.diagnostics.begin(), semantic.diagnostics.end(),
        [](const auto& item) { return item.code == "InvalidCallableArgumentTarget"; });
    require(count == 3,
            "non-assignable output, inout, and ref actuals should each be diagnosed");

    const auto call_start = source.find("capture(1'b1");
    const auto call_end = source.find(");", call_start);
    require(call_start != std::string::npos && call_end != std::string::npos,
            "test fixture is missing the expected task-call source span");
    for (const std::string_view expected_message : {
             "argument for output formal 'out_value' must be an assignable expression",
             "argument for inout formal 'io_value' must be an assignable expression",
             "argument for ref formal 'ref_value' must be an assignable expression"}) {
        const auto diagnostic = std::find_if(
            semantic.diagnostics.begin(), semantic.diagnostics.end(),
            [&](const auto& item) {
                return item.code == "InvalidCallableArgumentTarget" &&
                       item.message == expected_message;
            });
        require(diagnostic != semantic.diagnostics.end(),
                "non-assignable callable actual has an unexpected diagnostic code or message");
        require(diagnostic->location.source &&
                    diagnostic->location.byte_offset >= call_start &&
                    diagnostic->location.byte_offset < call_end &&
                    diagnostic->location.location.line == 11 &&
                    diagnostic->location.source->location(
                        diagnostic->location.byte_offset).line == 11,
                "callable actual diagnostic did not retain its call-site source location");
        const size_t formal_line =
            expected_message.find("output formal") != std::string_view::npos ? 2 : 3;
        require(diagnostic->related_location &&
                    diagnostic->related_location->location.line == formal_line,
                "callable actual diagnostic did not retain the formal declaration location");
    }
}

void test_callable_symbols_resolution_and_function_result_type() {
    const auto parsed = parse_verilog_syntax(
        "module typed_calls(input logic [3:0] data, output logic [7:0] result);\n"
        "  function automatic logic [7:0] extend(input logic [3:0] value);\n"
        "    extend = {4'b0, value};\n"
        "  endfunction\n"
        "  task consume(input logic [7:0] value);\n"
        "    begin end\n"
        "  endtask\n"
        "  assign result = extend(data);\n"
        "  always_comb consume(result);\n"
        "endmodule\n");
    require(!parsed.has_errors(), "valid function and task calls should parse");
    const auto semantic = analyze_verilog_semantics(parsed.syntax);
    require(!semantic.has_errors(), "valid callable declarations and calls should resolve");

    const auto* module_symbol =
        lookup_verilog_symbol(*semantic.model.compilation_unit_scope, "typed_calls", false);
    require(module_symbol && module_symbol->member_scope,
            "callable-containing module symbol or scope is missing");
    const auto* function_symbol =
        lookup_verilog_symbol(*module_symbol->member_scope, "extend", false);
    const auto* task_symbol =
        lookup_verilog_symbol(*module_symbol->member_scope, "consume", false);
    require(function_symbol && function_symbol->kind == VerilogSemanticSymbolKind::Function &&
                function_symbol->member_scope &&
                task_symbol && task_symbol->kind == VerilogSemanticSymbolKind::Task &&
                task_symbol->member_scope,
            "function/task semantic symbols or callable scopes are missing");
    const auto* function_argument =
        lookup_verilog_symbol(*function_symbol->member_scope, "value", false);
    const auto* task_argument =
        lookup_verilog_symbol(*task_symbol->member_scope, "value", false);
    require(function_argument &&
                function_argument->kind == VerilogSemanticSymbolKind::CallableArgument &&
                task_argument &&
                task_argument->kind == VerilogSemanticSymbolKind::CallableArgument,
            "callable argument symbols were not collected in callable scopes");
    require(function_symbol->callable.has_value(),
            "function symbol is missing callable metadata");
    require(function_symbol->callable->return_type == "logic" &&
                function_symbol->callable->return_packed_dimensions.size() == 1 &&
                function_symbol->callable->return_unpacked_dimensions.empty(),
            "function callable metadata lost its return type or dimensions");
    require(function_symbol->callable->formal_arguments.size() == 1 &&
                function_symbol->callable->formal_arguments.front() == function_argument,
            "function formal metadata is not an ordered list of formal symbols");
    require(function_argument->argument_direction == VerilogPortDirection::Input &&
                function_argument->declared_type == "logic" &&
                function_argument->packed_dimensions.size() == 1 &&
                function_argument->unpacked_dimensions.empty(),
            "function formal symbol lost direction, type, or dimensions");
    require(task_symbol->callable.has_value() &&
                task_symbol->callable->formal_arguments.size() == 1 &&
                task_symbol->callable->formal_arguments.front() == task_argument,
            "task formal metadata is not an ordered list of formal symbols");
    require(task_argument->argument_direction == VerilogPortDirection::Input &&
                task_argument->declared_type == "logic" &&
                task_argument->packed_dimensions.size() == 1 &&
                task_argument->unpacked_dimensions.empty(),
            "task formal symbol lost direction, type, or dimensions");

    const auto& module = parsed.syntax.modules.front();
    const auto* function_call = find_call(module.assignments.front().right.get(), "extend");
    require(function_call && function_call->operands.size() == 1,
            "function invocation is not represented by the expected call expression");
    const auto type_analysis = analyze_verilog_expression_types(parsed.syntax, semantic);
    const auto inferred = type_analysis.expression_types.find(function_call);
    require(inferred != type_analysis.expression_types.end() &&
                inferred->second.bit_width == 8,
            "function call expression did not infer its declared return width");
    require(std::any_of(semantic.references.begin(), semantic.references.end(),
                        [&](const auto& reference) {
                            return reference.name == "extend" &&
                                   reference.declaration == function_symbol;
                        }),
            "function call reference was not resolved to its callable symbol");
}

void test_argument_count_call_kind_and_unknown_call_diagnostics() {
    const auto parsed = parse_verilog_syntax(
        "module bad_calls(input logic [7:0] x, output logic [7:0] y);\n"
        "  function logic [7:0] f(input logic [7:0] a);\n"
        "    f = a;\n"
        "  endfunction\n"
        "  task t(input logic [7:0] a);\n"
        "    begin end\n"
        "  endtask\n"
        "  assign y = t(x);\n"
        "  always_comb begin\n"
        "    f();\n"
        "    t();\n"
        "    y = missing_callable(x);\n"
        "  end\n"
        "endmodule\n");
    require(!parsed.has_errors(), "call-diagnostic fixture should be syntactically valid");
    const auto semantic = analyze_verilog_semantics(parsed.syntax);
    require(has_callable_diagnostic(semantic, "argument") ||
                has_callable_diagnostic(semantic, "count") ||
                has_callable_diagnostic(semantic, "arity"),
            "wrong positional argument counts were not diagnosed");
    require(has_callable_diagnostic(semantic, "function", "task") ||
                has_callable_diagnostic(semantic, "task", "function"),
            "using a task as a function was not diagnosed");
    require(has_callable_diagnostic(semantic, "missing_callable") ||
                has_callable_diagnostic(semantic, "unresolved"),
            "unknown callable invocation was not diagnosed");

    const auto use_error = std::find_if(
        semantic.diagnostics.begin(), semantic.diagnostics.end(),
        [](const auto& diagnostic) {
            return diagnostic.location.location.line == 8 &&
                   (contains_case_insensitive(diagnostic.code, "call") ||
                    contains_case_insensitive(diagnostic.message, "call") ||
                    contains_case_insensitive(diagnostic.message, "task"));
        });
    require(use_error != semantic.diagnostics.end() && use_error->location.source &&
                use_error->location.byte_offset > 0,
            "function/task misuse diagnostic did not retain the invocation source location");
    require(std::any_of(semantic.diagnostics.begin(), semantic.diagnostics.end(),
                        [](const auto& diagnostic) {
                            return diagnostic.location.location.line == 10 ||
                                   diagnostic.location.location.line == 11;
                        }),
            "argument-count diagnostic did not identify a call site");
}

void test_argument_width_warning() {
    const auto parsed = parse_verilog_syntax(
        "module width_call(output logic [7:0] result);\n"
        "  function logic [7:0] narrow(input logic [3:0] value);\n"
        "    narrow = {4'b0, value};\n"
        "  endfunction\n"
        "  assign result = narrow(16'h1234);\n"
        "endmodule\n");
    require(!parsed.has_errors(), "width-conversion fixture should parse");
    const auto semantic = analyze_verilog_semantics(parsed.syntax);
    const auto types = analyze_verilog_expression_types(parsed.syntax, semantic);
    const auto warning = std::find_if(
        types.diagnostics.begin(), types.diagnostics.end(), [](const auto& diagnostic) {
            return diagnostic.severity == VerilogTypeDiagnosticSeverity::Warning &&
                   contains_case_insensitive(diagnostic.code + diagnostic.message, "width");
        });
    require(warning != types.diagnostics.end() && warning->location.source &&
                warning->location.location.line == 5,
            "known function-argument width conversion did not produce a located warning");
}

void test_named_callable_binding_integrated_with_semantics_and_types() {
    const auto parsed = parse_verilog_syntax(
        "module named_binding(input logic [3:0] nibble,\n"
        "                     output logic [7:0] result);\n"
        "  task automatic route(input logic [3:0] data,\n"
        "                      output logic [7:0] destination);\n"
        "    destination = {4'b0, data};\n"
        "  endtask\n"
        "  function automatic logic [7:0] choose(\n"
        "      input logic [3:0] narrow, input logic [7:0] wide);\n"
        "    choose = wide + narrow;\n"
        "  endfunction\n"
        "  always_comb begin\n"
        "    route(.destination(result), .data(nibble));\n"
        "    result = choose(.wide(result), .narrow(nibble));\n"
        "  end\n"
        "endmodule\n");
    require(!parsed.has_errors(), "named callable binding fixture should parse cleanly");
    const auto semantic = analyze_verilog_semantics(parsed.syntax);
    require(!semantic.has_errors(),
            "semantic analysis rejected named task actuals or output lvalues");
    const auto types = analyze_verilog_expression_types(parsed.syntax, semantic);
    require(!types.has_errors(),
            "type analysis rejected named function/task argument bindings");
    require(std::none_of(types.diagnostics.begin(), types.diagnostics.end(),
                         [](const auto& diagnostic) {
                             return diagnostic.severity ==
                                    VerilogTypeDiagnosticSeverity::Warning;
                         }),
            "named actuals were compared positionally instead of by formal name");
}

void test_macro_origin_and_package_qualified_function_call() {
    const auto macro_parsed = parse_verilog_syntax(
        "`define UNKNOWN_CALL absent_callable\n"
        "module macro_call(output logic [7:0] result);\n"
        "  assign result = `UNKNOWN_CALL(8'h1);\n"
        "endmodule\n");
    require(!macro_parsed.has_errors(), "macro call fixture should parse");
    const auto macro_semantic = analyze_verilog_semantics(macro_parsed.syntax);
    const auto macro_diagnostic = std::find_if(
        macro_semantic.diagnostics.begin(), macro_semantic.diagnostics.end(),
        [](const auto& diagnostic) {
            return contains_case_insensitive(diagnostic.code + diagnostic.message,
                                             "absent_callable") ||
                   contains_case_insensitive(diagnostic.code + diagnostic.message,
                                             "unresolved");
        });
    require(macro_diagnostic != macro_semantic.diagnostics.end() &&
                macro_diagnostic->location.origin_kind ==
                    VerilogPreprocessedOriginKind::MacroExpansion &&
                !macro_diagnostic->location.expansion_stack.empty() &&
                macro_diagnostic->location.expansion_stack.back().macro_name ==
                    "UNKNOWN_CALL",
            "unknown callable diagnostic lost its macro expansion origin");

    const auto package_parsed = parse_verilog_syntax(
        "package callable_functions;\n"
        "  function automatic logic [7:0] identity(input logic [7:0] value);\n"
        "    identity = value;\n"
        "  endfunction\n"
        "endpackage\n"
        "module package_user(input logic [7:0] x, output logic [7:0] y);\n"
        "  assign y = callable_functions::identity(x);\n"
        "endmodule\n");
    require(!package_parsed.has_errors(), "package-qualified call should parse");
    const auto package_semantic = analyze_verilog_semantics(package_parsed.syntax);
    require(!package_semantic.has_errors(),
            "package-qualified function call should resolve without diagnostics");
    require(std::any_of(package_semantic.references.begin(),
                        package_semantic.references.end(), [](const auto& reference) {
                            return reference.declaration &&
                                   reference.declaration->name == "identity" &&
                                   contains_case_insensitive(reference.name, "identity");
                        }),
            "package-qualified function reference did not resolve to the package callable");
}

void test_slang_accepts_supported_callable_syntax() {
    const std::string source =
        "package ref_pkg;\n"
        "  function automatic logic [7:0] id(input logic [7:0] value);\n"
        "    id = value;\n"
        "  endfunction\n"
        "endpackage\n"
        "module callable_reference(input logic [7:0] data, output logic [7:0] result);\n"
        "  function automatic logic [7:0] add_one(input logic [7:0] value);\n"
        "    add_one = value + 1'b1;\n"
        "  endfunction\n"
        "  task automatic capture(input logic [7:0] value, output logic done);\n"
        "    begin done = value[0]; end\n"
        "  endtask\n"
        "  logic done;\n"
        "  assign result = ref_pkg::id(add_one(data));\n"
        "  always_comb capture(result, done);\n"
        "endmodule\n";
    slang::SourceManager source_manager;
    const auto tree =
        slang::syntax::SyntaxTree::fromText(source, source_manager, "callables.sv");
    require(tree && tree->diagnostics().empty(),
            "Slang rejected the supported callable syntax acceptance fixture");
}

} // namespace

int main() {
    try {
        test_declarations_bodies_and_locations();
        test_nonassignable_output_inout_ref_actual_diagnostics();
        test_callable_symbols_resolution_and_function_result_type();
        test_argument_count_call_kind_and_unknown_call_diagnostics();
        test_argument_width_warning();
        test_named_callable_binding_integrated_with_semantics_and_types();
        test_macro_origin_and_package_qualified_function_call();
        test_slang_accepts_supported_callable_syntax();
    }
    catch (const std::exception& error) {
        std::cerr << "EDA Verilog callable test failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
