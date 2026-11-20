#include "labneura/eda/verilog_native_lowering.h"
#include "labneura/eda/native_verilog_project_context.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {
using namespace labneura::eda;
using namespace labneura::eda::detail;

void require(bool condition, std::string_view message) {
    if (!condition)
        throw std::runtime_error(std::string(message));
}

const VerilogSemanticScope& module_scope(const NativeVerilogProjectContext& context,
                                         std::string_view name) {
    const auto& root = *context.elaboration->semantic.model.compilation_unit_scope;
    for (const auto& scope : root.children)
        if (scope->kind == VerilogSemanticScopeKind::Module && scope->name == name)
            return *scope;
    throw std::runtime_error("missing module semantic scope");
}

void test_retains_multiple_sources_and_identity_for_downstream_analysis() {
    auto context = build_native_verilog_project_context(
        {{"module helper(input logic a, output logic y); assign y = a; endmodule\n",
          "helper.sv"},
         {"module top #(parameter int WIDTH = 4)(input logic [WIDTH-1:0] a, b, "
          "output logic [WIDTH-1:0] y);\n"
          "  assign y = a & b;\n"
          "endmodule\n",
          "top.sv"}},
        {}, {}, VerilogElaborationOptions{.top_module = "top"});

    require(context->ready(), "valid multi-source native project was not ready");
    require(context->source_files.size() == 2 &&
                context->parsed.syntax.modules.size() == 2,
            "native project lost source or module ownership");
    require(context->parsed.syntax.modules[0].location.source->name() == "helper.sv" &&
                context->parsed.syntax.modules[1].location.source->name() == "top.sv",
            "syntax locations did not retain independent source buffers");
    require(context->parsed.syntax.modules[1].location.source->text().find(
                "parameter int WIDTH = 4") != std::string_view::npos,
            "source text did not outlive the input source units");

    const auto& scope = module_scope(*context, "top");
    const auto* port = lookup_verilog_symbol(scope, "a", false);
    require(port && port->kind == VerilogSemanticSymbolKind::Port,
            "native semantic symbol identity was unavailable");
    const auto port_type = context->types->symbol_types.find(port);
    require(port_type != context->types->symbol_types.end() &&
                port_type->second.bit_width == std::optional<size_t>(4),
            "type analysis did not retain the parameterized port symbol type");
    require(context->elaboration->top &&
                context->elaboration->top->module_symbol ==
                    lookup_verilog_symbol(
                        *context->elaboration->semantic.model.compilation_unit_scope,
                        "top", false),
            "elaborated top did not retain the semantic module identity");

    const auto parameter = context->elaboration->top->parameters.find("WIDTH");
    require(parameter != context->elaboration->top->parameters.end() &&
                parameter->second.to_uint64() == std::optional<uint64_t>(4),
            "elaborated parameter environment was not retained");

    auto lowered = lower_verilog_native(
        context->parsed.syntax, *context->elaboration, *context->types,
        *context->eligibility, "top");
    require(lowered.succeeded() && lowered.module->get_name() == "top",
            "retained context could not be consumed by native lowering");
}

void test_invalid_frontend_stops_before_dependent_analysis() {
    auto context = build_native_verilog_project_context(
        {{"module broken(input logic a; endmodule\n", "broken.sv"}},
        {}, {}, VerilogElaborationOptions{.top_module = "broken"});
    require(!context->ready() && context->parsed.has_errors(),
            "invalid native project was reported ready");
    require(!context->elaboration && !context->types && !context->eligibility,
            "dependent analysis ran after syntax failure");
}

} // namespace

int main() {
    try {
        test_retains_multiple_sources_and_identity_for_downstream_analysis();
        test_invalid_frontend_stops_before_dependent_analysis();
    } catch (const std::exception& error) {
        std::cerr << "native retained project context test failed: " << error.what()
                  << '\n';
        return 1;
    }
    return 0;
}
