#include "labneura/eda/verilog_task_execution.h"

#include "labneura/eda/verilog_syntax.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace labneura::eda;

void require(bool condition, std::string_view message) {
    if (!condition)
        throw std::runtime_error(std::string(message));
}

VerilogConstantValue value(uint64_t number, size_t width = 8) {
    VerilogConstantValue result;
    result.bit_width = width;
    result.bits.resize(width, VerilogLogicBit::Zero);
    for (size_t bit = 0; bit < width && bit < 64; ++bit)
        if ((number >> bit) & 1U)
            result.bits[bit] = VerilogLogicBit::One;
    return result;
}

uint64_t known_value(const VerilogConstantEnvironment& environment,
                     std::string_view name) {
    const auto found = environment.find(std::string(name));
    require(found != environment.end(), "expected caller environment value is missing");
    const auto converted = found->second.to_uint64();
    require(converted.has_value(),
            "expected caller environment value is unknown for " + std::string(name));
    return *converted;
}

const VerilogExpression* find_task_call(const VerilogStatement* statement,
                                        std::string_view task_name) {
    if (!statement)
        return nullptr;
    if (statement->kind == VerilogStatementKind::TaskCall &&
        statement->call_expression && statement->call_expression->first &&
        statement->call_expression->first->text == task_name)
        return statement->call_expression.get();
    if (const auto* found = find_task_call(statement->first_statement.get(), task_name))
        return found;
    if (const auto* found = find_task_call(statement->second_statement.get(), task_name))
        return found;
    for (const auto& nested : statement->statements)
        if (const auto* found = find_task_call(nested.get(), task_name))
            return found;
    for (const auto& item : statement->case_items)
        if (const auto* found = find_task_call(item.statement.get(), task_name))
            return found;
    return nullptr;
}

struct Fixture {
    VerilogSyntaxParseResult parsed;
    VerilogSemanticAnalysisResult semantic;
    VerilogTypeAnalysisResult types;

    explicit Fixture(std::string source)
        : parsed(parse_verilog_syntax(std::move(source))),
          semantic(analyze_verilog_semantics(parsed.syntax)),
          types(analyze_verilog_expression_types(parsed.syntax, semantic)) {
        require(!parsed.has_errors(), "delay execution fixture failed to parse");
        require(!semantic.has_errors(), "delay execution fixture failed semantic analysis");
    }

    const VerilogExpression& call(std::string_view name) const {
        for (const auto& module : parsed.syntax.modules)
            for (const auto& block : module.procedural_blocks)
                if (const auto* found = find_task_call(block.body.get(), name))
                    return *found;
        throw std::runtime_error("task call not found in delay fixture");
    }
};

bool has_code(const VerilogTaskExecutionResult& result, std::string_view code) {
    return std::any_of(result.diagnostics.begin(), result.diagnostics.end(),
                       [&](const auto& diagnostic) {
                           return diagnostic.code == code;
                       });
}

void test_zero_and_positive_delay_ordering() {
    Fixture fixture(
        "module delay_ordering;\n"
        "  logic [7:0] result;\n"
        "  task automatic ordered(output logic [7:0] out_value);\n"
        "    begin\n"
        "      out_value = 8'd1;\n"
        "      #0 out_value = out_value + 8'd1;\n"
        "      #3 out_value = out_value + 8'd1;\n"
        "    end\n"
        "  endtask\n"
        "  initial ordered(result);\n"
        "endmodule\n");
    VerilogConstantEnvironment environment{{"result", value(77)}};
    const auto result = execute_verilog_task_call(
        fixture.parsed.syntax, fixture.semantic, fixture.types,
        fixture.call("ordered"), environment);
    require(result.completed && result.status == VerilogTaskExecutionStatus::Completed &&
                result.diagnostics.empty(),
            "zero and positive procedural delays did not execute");
    require(result.elapsed_delay_ticks == 3,
            "zero delay advanced time or positive delay accounting was incorrect");
    require(known_value(environment, "result") == 3,
            "statements around delays did not preserve source order");
}

void test_delay_expression_formal_and_nested_task_accounting() {
    Fixture fixture(
        "module delay_expression;\n"
        "  logic [7:0] result;\n"
        "  task automatic child(input logic [7:0] ticks,\n"
        "                      inout logic [7:0] out_value);\n"
        "    #(ticks + 8'd2) out_value = out_value + 8'd1;\n"
        "  endtask\n"
        "  task automatic parent(input logic [7:0] ticks,\n"
        "                       output logic [7:0] out_value);\n"
        "    begin\n"
        "      out_value = 8'd4;\n"
        "      #ticks child(ticks, out_value);\n"
        "    end\n"
        "  endtask\n"
        "  initial parent(8'd3, result);\n"
        "endmodule\n");
    VerilogConstantEnvironment environment{{"result", value(0)}};
    const auto result = execute_verilog_task_call(
        fixture.parsed.syntax, fixture.semantic, fixture.types,
        fixture.call("parent"), environment);
    require(result.completed && result.status == VerilogTaskExecutionStatus::Completed &&
                result.diagnostics.empty(),
            "formal-dependent or nested task delays failed");
    require(result.elapsed_delay_ticks == 8 &&
                known_value(environment, "result") == 5,
            "delay expressions or shared nested elapsed-time accounting were incorrect");
}

void test_invalid_delay_values_fail_with_locations_and_rollback() {
    Fixture fixture(
        "module invalid_delays;\n"
        "  logic [7:0] result;\n"
        "  task automatic negative(output logic [7:0] out_value);\n"
        "    #(-1) out_value = 8'd1;\n"
        "  endtask\n"
        "  task automatic unknown(output logic [7:0] out_value);\n"
        "    #(1'bx) out_value = 8'd2;\n"
        "  endtask\n"
        "  task automatic too_wide(output logic [7:0] out_value);\n"
        "    #(65'h1_0000000000000000) out_value = 8'd3;\n"
        "  endtask\n"
        "  task automatic unsupported(output logic [7:0] out_value);\n"
        "    #(\"delay\") out_value = 8'd4;\n"
        "  endtask\n"
        "  initial begin\n"
        "    negative(result); unknown(result); too_wide(result); unsupported(result);\n"
        "  end\n"
        "endmodule\n");

    for (const std::string_view task_name : {"negative", "unknown", "too_wide",
                                             "unsupported"}) {
        VerilogConstantEnvironment environment{{"result", value(99)}};
        const auto result = execute_verilog_task_call(
            fixture.parsed.syntax, fixture.semantic, fixture.types,
            fixture.call(task_name), environment);
        const auto expected_code = task_name == "unsupported"
                                       ? "UnsupportedCallableDelayExpression"
                                       : "InvalidCallableDelayValue";
        require(!result.completed && result.status == VerilogTaskExecutionStatus::Failed &&
                    has_code(result, expected_code) &&
                    !result.diagnostics.empty() &&
                    result.diagnostics.front().location.source &&
                    result.elapsed_delay_ticks == 0 &&
                    known_value(environment, "result") == 99,
                "invalid delay must fail at its source and preserve caller state");
    }
}

void test_delay_accumulation_overflow_is_diagnosed() {
    Fixture fixture(
        "module overflowing_delay;\n"
        "  logic [7:0] result;\n"
        "  task automatic overflow(output logic [7:0] out_value);\n"
        "    begin\n"
        "      #64'hffffffffffffffff;\n"
        "      #1 out_value = 8'd1;\n"
        "    end\n"
        "  endtask\n"
        "  initial overflow(result);\n"
        "endmodule\n");
    VerilogConstantEnvironment environment{{"result", value(55)}};
    const auto result = execute_verilog_task_call(
        fixture.parsed.syntax, fixture.semantic, fixture.types,
        fixture.call("overflow"), environment);
    require(!result.completed && result.status == VerilogTaskExecutionStatus::Failed &&
                has_code(result, "CallableDelayOverflow") &&
                result.elapsed_delay_ticks == std::numeric_limits<uint64_t>::max() &&
                known_value(environment, "result") == 55,
            "virtual-time overflow must diagnose without committing task outputs");
}

} // namespace

int main() {
    try {
        test_zero_and_positive_delay_ordering();
        test_delay_expression_formal_and_nested_task_accounting();
        test_invalid_delay_values_fail_with_locations_and_rollback();
        test_delay_accumulation_overflow_is_diagnosed();
    }
    catch (const std::exception& error) {
        std::cerr << "EDA Verilog delay execution test failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
