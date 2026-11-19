#include "labneura/eda/verilog_task_execution.h"

#include "labneura/eda/verilog_syntax.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
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
    for (size_t bit = 0; bit < width; ++bit)
        if ((number >> bit) & 1)
            result.bits[bit] = VerilogLogicBit::One;
    return result;
}

uint64_t known_value(const VerilogConstantEnvironment& environment,
                     std::string_view name) {
    const auto found = environment.find(std::string(name));
    require(found != environment.end(), "expected caller environment value is missing");
    const auto converted = found->second.to_uint64();
    require(converted.has_value(), "expected caller environment value is unknown");
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
    if (const auto* found =
            find_task_call(statement->loop_initialization.get(), task_name))
        return found;
    if (const auto* found = find_task_call(statement->loop_step.get(), task_name))
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
        require(!parsed.has_errors(), "task execution fixture failed to parse");
        if (semantic.has_errors())
            throw std::runtime_error("task execution fixture failed semantic analysis: " +
                                     semantic.diagnostics.front().code + ": " +
                                     semantic.diagnostics.front().message);
    }

    const VerilogExpression& call(std::string_view name) const {
        for (const auto& module : parsed.syntax.modules)
            for (const auto& block : module.procedural_blocks)
                if (const auto* found = find_task_call(block.body.get(), name))
                    return *found;
        throw std::runtime_error("task call not found in fixture");
    }
};

bool has_code(const VerilogTaskExecutionResult& result, std::string_view code) {
    return std::any_of(result.diagnostics.begin(), result.diagnostics.end(),
                       [&](const auto& diagnostic) {
                           return diagnostic.code == code;
                       });
}

void test_sequential_flow_and_bounded_loops() {
    Fixture fixture(
        "module flow(output logic [7:0] result);\n"
        "  task automatic exercise(input logic [3:0] n, output logic [7:0] out);\n"
        "    integer i;\n"
        "    begin\n"
        "      out = 0;\n"
        "      for (i = 0; i < n; i = i + 1) begin\n"
        "        if (i == 2) out = out + 2; else out = out + 1;\n"
        "      end\n"
        "      case (n)\n"
        "        3: out = out + 1;\n"
        "        default: out = out + 9;\n"
        "      endcase\n"
        "      repeat (1) out = out + 1;\n"
        "      while (out < 8'd7) out = out + 1;\n"
        "      forever begin\n"
        "        out = out + 1;\n"
        "        if (out == 8'd9) return;\n"
        "      end\n"
        "      out = 0;\n"
        "    end\n"
        "  endtask\n"
        "  initial begin exercise(4'd3, result); end\n"
        "endmodule\n");
    VerilogConstantEnvironment environment{{"result", value(42)}};
    const auto result = execute_verilog_task_call(
        fixture.parsed.syntax, fixture.semantic, fixture.types,
        fixture.call("exercise"), environment);
    require(result.completed && result.diagnostics.empty(),
            "task flow did not complete successfully");
    require(known_value(environment, "result") == 9,
            "blocking assignments, branches, case, loops, or return were mis-executed");
    require(!environment.contains("i"),
            "task-local variables leaked into the caller environment");
    require(result.executed_steps > 0,
            "task execution did not count executed statement steps");
}

void test_casez_and_casex_item_wildcards() {
    Fixture fixture(
        "module case_wildcards;\n"
        "  task automatic casez_match(input logic [3:0] selector,\n"
        "                            output logic [7:0] result);\n"
        "    casez (selector)\n"
        "      4'b1z01: result = 8'd1;\n"
        "      default: result = 8'd0;\n"
        "    endcase\n"
        "  endtask\n"
        "  task automatic casex_match(input logic [3:0] selector,\n"
        "                            output logic [7:0] result);\n"
        "    casex (selector)\n"
        "      4'b1x01: result = 8'd2;\n"
        "      default: result = 8'd0;\n"
        "    endcase\n"
        "  endtask\n"
        "  task automatic casez_x_is_not_wild(input logic [3:0] selector,\n"
        "                                     output logic [7:0] result);\n"
        "    casez (selector)\n"
        "      4'b1x01: result = 8'd3;\n"
        "      default: result = 8'd0;\n"
        "    endcase\n"
        "  endtask\n"
        "  logic [7:0] z_result;\n"
        "  logic [7:0] x_result;\n"
        "  logic [7:0] exact_x_result;\n"
        "  initial begin\n"
        "    casez_match(4'b1101, z_result);\n"
        "    casex_match(4'b1101, x_result);\n"
        "    casez_x_is_not_wild(4'b1101, exact_x_result);\n"
        "  end\n"
        "endmodule\n");
    VerilogConstantEnvironment environment{{"z_result", value(0)},
                                           {"x_result", value(0)},
                                           {"exact_x_result", value(0)}};
    for (const auto& [task, result_name] :
         std::vector<std::pair<std::string_view, std::string_view>>{
             {"casez_match", "z_result"},
             {"casex_match", "x_result"},
             {"casez_x_is_not_wild", "exact_x_result"}}) {
        const auto result = execute_verilog_task_call(
            fixture.parsed.syntax, fixture.semantic, fixture.types,
            fixture.call(task), environment);
        require(result.completed && result.diagnostics.empty(),
                "casez/casex item pattern failed to execute");
        const uint64_t expected = result_name == "z_result" ? 1 :
                                  result_name == "x_result" ? 2 : 0;
        require(known_value(environment, result_name) == expected,
                "casez/casex item wildcard matching was incorrect");
    }
}

void test_nested_calls_formal_propagation_and_function_expressions() {
    Fixture fixture(
        "module propagation;\n"
        "  function automatic logic [7:0] increment(input logic [7:0] x);\n"
        "    increment = x + 1;\n"
        "  endfunction\n"
        "  task automatic child(input logic [7:0] x,\n"
        "                      output logic [7:0] y,\n"
        "                      inout logic [7:0] z);\n"
        "    begin\n"
        "      y = increment(x) + increment(8'd1);\n"
        "      z = z + 2;\n"
        "    end\n"
        "  endtask\n"
        "  task automatic parent(input logic [7:0] x,\n"
        "                       output logic [7:0] y,\n"
        "                       inout logic [7:0] z);\n"
        "    child(x, y, z);\n"
        "  endtask\n"
        "  initial begin parent(8'd4, result, io_value); end\n"
        "  logic [7:0] result;\n"
        "  logic [7:0] io_value;\n"
        "endmodule\n");
    VerilogConstantEnvironment environment{{"result", value(77)},
                                           {"io_value", value(10)}};
    const auto result = execute_verilog_task_call(
        fixture.parsed.syntax, fixture.semantic, fixture.types,
        fixture.call("parent"), environment);
    require(result.completed && result.diagnostics.empty(),
            "nested task/function calls or formal propagation failed");
    require(known_value(environment, "result") == 7 &&
                known_value(environment, "io_value") == 12,
            "input/output/inout bindings did not propagate expected values");
}

void test_known_width_conversion_on_input_and_writeback() {
    Fixture fixture(
        "module width_conversion(output logic [7:0] out_value,\n"
        "                        output logic [7:0] io_value);\n"
        "  task automatic narrow(input logic [3:0] input_value,\n"
        "                       output logic [3:0] output_value,\n"
        "                       inout logic [3:0] inout_value);\n"
        "    begin\n"
        "      output_value = input_value;\n"
        "      inout_value = inout_value + 1'b1;\n"
        "    end\n"
        "  endtask\n"
        "  initial narrow(8'd17, out_value, io_value);\n"
        "endmodule\n");
    VerilogConstantEnvironment environment{{"out_value", value(0)},
                                           {"io_value", value(0xfe)}};
    const auto result = execute_verilog_task_call(
        fixture.parsed.syntax, fixture.semantic, fixture.types,
        fixture.call("narrow"), environment);
    require(result.completed && known_value(environment, "out_value") == 1 &&
                known_value(environment, "io_value") == 15,
            "known width conversions did not truncate input and extend writeback");
}

void test_declared_for_variable_has_loop_lifetime() {
    Fixture fixture(
        "module loop_scope;\n"
        "  integer i;\n"
        "  task automatic check(output logic [31:0] result);\n"
        "    begin\n"
        "      for (integer i = 0; i < 1; i = i + 1) ;\n"
        "      result = i;\n"
        "    end\n"
        "  endtask\n"
        "  initial check(result);\n"
        "  logic [31:0] result;\n"
        "endmodule\n");
    VerilogConstantEnvironment environment{{"i", value(7, 32)},
                                           {"result", value(0, 32)}};
    const auto result = execute_verilog_task_call(
        fixture.parsed.syntax, fixture.semantic, fixture.types,
        fixture.call("check"), environment);
    require(result.completed && known_value(environment, "result") == 7,
            "for-loop declaration leaked past its loop scope");
}

void test_output_formal_starts_uninitialized() {
    Fixture fixture(
        "module output_default;\n"
        "  task automatic leave_unassigned(output logic out_value,\n"
        "                                  output bit zero_value);\n"
        "    begin end\n"
        "  endtask\n"
        "  logic out_value;\n"
        "  bit zero_value;\n"
        "  initial leave_unassigned(out_value, zero_value);\n"
        "endmodule\n");
    VerilogConstantEnvironment environment{{"out_value", value(1, 1)},
                                           {"zero_value", value(1, 1)}};
    const auto result = execute_verilog_task_call(
        fixture.parsed.syntax, fixture.semantic, fixture.types,
        fixture.call("leave_unassigned"), environment);
    require(result.completed, "unassigned output formal should still complete");
    const auto found = environment.find("out_value");
    require(found != environment.end() && found->second.bit_width == 1 &&
                found->second.bits == std::vector<VerilogLogicBit>{VerilogLogicBit::X},
            "output formal did not start at its declared 4-state X default");
    const auto zero = environment.find("zero_value");
    require(zero != environment.end() &&
                zero->second.bits == std::vector<VerilogLogicBit>{VerilogLogicBit::Zero},
            "output formal did not start at its declared 2-state zero default");
}

void test_unsupported_procedural_statements_are_diagnosed() {
    Fixture fixture(
        "module unsupported_calls(input logic clk, input logic flag,\n"
        "                         output logic [7:0] result);\n"
        "  task automatic nonblocking(output logic [7:0] out_value);\n"
        "    out_value <= 8'd1;\n"
        "  endtask\n"
        "  task automatic delayed(output logic [7:0] out_value);\n"
        "    #1 out_value = 8'd2;\n"
        "  endtask\n"
        "  task automatic evented(input logic clock,\n"
        "                        output logic [7:0] out_value);\n"
        "    @(posedge clock) out_value = 8'd3;\n"
        "  endtask\n"
        "  task automatic wildcard_event(output logic [7:0] out_value);\n"
        "    @* out_value = 8'd5;\n"
        "  endtask\n"
        "  task automatic waiting(input logic condition,\n"
        "                        output logic [7:0] out_value);\n"
        "    wait (condition) out_value = 8'd4;\n"
        "  endtask\n"
        "  task automatic unsupported(output logic [7:0] out_value);\n"
        "    begin end\n"
        "  endtask\n"
        "  initial begin\n"
        "    nonblocking(result);\n"
        "    delayed(result);\n"
        "    evented(clk, result);\n"
        "    wildcard_event(result);\n"
        "    waiting(flag, result);\n"
        "    unsupported(result);\n"
        "  end\n"
        "endmodule\n");
    auto& declarations = fixture.parsed.syntax.modules.front().callables;
    const auto unsupported_decl =
        std::find_if(declarations.begin(), declarations.end(), [](const auto& task) {
            return task.name == "unsupported";
        });
    require(unsupported_decl != declarations.end() && unsupported_decl->body,
            "fixture is missing its unsupported-statement task");
    unsupported_decl->body->kind = VerilogStatementKind::Unsupported;
    unsupported_decl->body->text = "unsupported procedural form";

    const auto check = [&](std::string_view task_name, std::string_view code) {
        VerilogConstantEnvironment environment{{"clk", value(0, 1)},
                                               {"flag", value(1, 1)},
                                               {"result", value(99)}};
        const auto result = execute_verilog_task_call(
            fixture.parsed.syntax, fixture.semantic, fixture.types,
            fixture.call(task_name), environment);
        require(!result.completed && has_code(result, code) &&
                    known_value(environment, "result") == 99,
                "unsupported task statement was ignored or mutated caller state");
    };
    check("nonblocking", "UnsupportedNonblockingCallableAssignment");
    check("evented", "NoMatchingEvent");
    check("wildcard_event", "UnsupportedCallableWildcardEventControl");
    check("waiting", "UnsupportedCallableWait");
    check("unsupported", "UnsupportedCallableStatement");

    VerilogConstantEnvironment delay_environment{{"clk", value(0, 1)},
                                                 {"flag", value(1, 1)},
                                                 {"result", value(99)}};
    const auto delayed = execute_verilog_task_call(
        fixture.parsed.syntax, fixture.semantic, fixture.types,
        fixture.call("delayed"), delay_environment);
    require(delayed.completed && delayed.status == VerilogTaskExecutionStatus::Completed &&
                delayed.elapsed_delay_ticks == 1 &&
                known_value(delay_environment, "result") == 2,
            "supported positive delay did not execute in order with virtual time");
}

void test_recursion_call_depth_and_step_limits_are_transactional() {
    Fixture fixture(
        "module limits;\n"
        "  task automatic recurse(input logic [7:0] n,\n"
        "                        output logic [7:0] out_value);\n"
        "    if (n == 0) out_value = 8'd1;\n"
        "    else recurse(n - 1, out_value);\n"
        "  endtask\n"
        "  task automatic spin(inout logic [7:0] value);\n"
        "    repeat (100) value = value + 1;\n"
        "  endtask\n"
        "  initial begin recurse(8'd2, result); spin(count); end\n"
        "  logic [7:0] result;\n"
        "  logic [7:0] count;\n"
        "endmodule\n");

    VerilogConstantEnvironment recursive_environment{{"result", value(55)}};
    const auto recursive = execute_verilog_task_call(
        fixture.parsed.syntax, fixture.semantic, fixture.types,
        fixture.call("recurse"), recursive_environment);
    const auto* recursive_call =
        find_task_call(fixture.parsed.syntax.modules.front().callables.front().body.get(),
                       "recurse");
    const auto recursion_diagnostic =
        std::find_if(recursive.diagnostics.begin(), recursive.diagnostics.end(),
                     [](const auto& diagnostic) {
                         return diagnostic.code == "RecursiveCallableExecution";
                     });
    require(!recursive.completed && has_code(recursive, "RecursiveCallableExecution") &&
                known_value(recursive_environment, "result") == 55 &&
                recursive_call && recursion_diagnostic != recursive.diagnostics.end() &&
                recursion_diagnostic->location.byte_offset ==
                    recursive_call->location.byte_offset &&
                recursion_diagnostic->related_location.has_value(),
            "recursive task execution was not diagnosed transactionally");

    VerilogCallableExecutionOptions depth_options;
    depth_options.maximum_call_depth = 1;
    VerilogCallableExecutionState depth_state(depth_options);
    Fixture depth_fixture(
        "module call_depth;\n"
        "  task automatic inner(output logic [7:0] value);\n"
        "    value = 8'd1;\n"
        "  endtask\n"
        "  task automatic outer(output logic [7:0] value);\n"
        "    inner(value);\n"
        "  endtask\n"
        "  logic [7:0] result;\n"
        "  initial outer(result);\n"
        "endmodule\n");
    VerilogConstantEnvironment depth_environment{{"result", value(66)}};
    const auto depth = execute_verilog_task_call(
        depth_fixture.parsed.syntax, depth_fixture.semantic, depth_fixture.types,
        depth_fixture.call("outer"), depth_environment, depth_state);
    const auto* inner_call =
        find_task_call(depth_fixture.parsed.syntax.modules.front().callables.back().body.get(),
                       "inner");
    const auto depth_diagnostic =
        std::find_if(depth.diagnostics.begin(), depth.diagnostics.end(),
                     [](const auto& diagnostic) {
                         return diagnostic.code == "CallableExecutionDepthExceeded";
                     });
    require(!depth.completed && has_code(depth, "CallableExecutionDepthExceeded") &&
                known_value(depth_environment, "result") == 66 && inner_call &&
                depth_diagnostic != depth.diagnostics.end() &&
                depth_diagnostic->location.byte_offset == inner_call->location.byte_offset &&
                depth_diagnostic->related_location.has_value(),
            "shared call-depth limit was not enforced transactionally");

    VerilogCallableExecutionOptions step_options;
    step_options.maximum_statement_steps = 4;
    VerilogCallableExecutionState step_state(step_options);
    VerilogConstantEnvironment step_environment{{"count", value(7)}};
    const auto step = execute_verilog_task_call(
        fixture.parsed.syntax, fixture.semantic, fixture.types,
        fixture.call("spin"), step_environment, step_state);
    require(!step.completed &&
                has_code(step, "CallableExecutionStepLimitExceeded") &&
                known_value(step_environment, "count") == 7,
            "shared statement-step limit was not enforced transactionally");
}

void test_named_binding_order_and_task_local_isolation() {
    Fixture fixture(
        "module named_execution;\n"
        "  logic [7:0] input_value, output_value, io_value;\n"
        "  task automatic compute(input logic [7:0] input_value,\n"
        "      output logic [7:0] output_value, inout logic [7:0] io_value);\n"
        "    logic [7:0] local_value = input_value;\n"
        "    begin\n"
        "      output_value = local_value + 1;\n"
        "      io_value = io_value + local_value;\n"
        "      local_value = 8'hff;\n"
        "    end\n"
        "  endtask\n"
        "  initial begin\n"
        "    compute(.io_value(io_value), .output_value(output_value),\n"
        "            .input_value(input_value));\n"
        "  end\n"
        "endmodule\n");
    VerilogConstantEnvironment environment{{"input_value", value(7)},
                                           {"output_value", value(91)},
                                           {"io_value", value(10)}};
    for (int invocation = 0; invocation < 2; ++invocation) {
        const auto result = execute_verilog_task_call(
            fixture.parsed.syntax, fixture.semantic, fixture.types,
            fixture.call("compute"), environment);
        require(result.completed && result.diagnostics.empty(),
                "named task arguments failed during execution");
        require(known_value(environment, "output_value") == 8 &&
                    known_value(environment, "io_value") == (invocation == 0 ? 17 : 24),
                "named actual order or task local initialization was incorrect");
    }
}

void test_selected_and_concatenated_writeback_and_inout() {
    Fixture fixture(
        "module selected_writeback;\n"
        "  task automatic set(output logic [7:0] out, inout logic [3:0] io);\n"
        "    logic [7:0] local_value;\n"
        "    begin\n"
        "      local_value = 0;\n"
        "      {local_value[7:4], local_value[3:0]} = 8'hA5;\n"
        "      local_value[7:4] = local_value[3:0];\n"
        "      out = local_value;\n"
        "      io = io + 1;\n"
        "    end\n"
        "  endtask\n"
        "  logic [3:0] high;\n"
        "  logic [3:0] low;\n"
        "  logic [7:0] packed_value;\n"
        "  initial begin set({high, low}, packed_value[7:4]); end\n"
        "endmodule\n");
    VerilogConstantEnvironment environment{{"high", value(0, 4)},
                                           {"low", value(0, 4)},
                                           {"packed_value", value(0xC0, 8)}};
    const auto result = execute_verilog_task_call(
        fixture.parsed.syntax, fixture.semantic, fixture.types,
        fixture.call("set"), environment);
    require(result.completed && result.diagnostics.empty(),
            "selected/concatenated output and inout writeback should complete");
    require(known_value(environment, "high") == 5 &&
                known_value(environment, "low") == 5 &&
                known_value(environment, "packed_value") == 0xD0,
            "concatenated output split or selected inout copy-in/out was incorrect");
}

void test_overlapping_output_writeback_is_diagnosed_transactionally() {
    Fixture fixture(
        "module unsupported_writeback;\n"
        "  task automatic set_bit(output logic bit_value);\n"
        "    bit_value = 1'b1;\n"
        "  endtask\n"
        "  logic [3:0] result;\n"
        "  initial set_bit({result[1:0], result[1]});\n"
        "endmodule\n");
    VerilogConstantEnvironment environment{{"result", value(10, 4)}};
    const auto result = execute_verilog_task_call(
        fixture.parsed.syntax, fixture.semantic, fixture.types,
        fixture.call("set_bit"), environment);
    const auto diagnostic = std::find_if(
        result.diagnostics.begin(), result.diagnostics.end(), [](const auto& item) {
            return item.code == "OverlappingAssignmentDestinations";
        });
    require(!result.completed && diagnostic != result.diagnostics.end() &&
                diagnostic->location.source &&
                diagnostic->location.location.line == 6 &&
                diagnostic->related_location.has_value() &&
                known_value(environment, "result") == 10,
            "overlapping output writeback must fail with source locations and no caller mutation");
}

void test_ref_execution_does_not_fall_back_to_copy_semantics() {
    Fixture fixture(
        "module ref_execution;\n"
        "  task automatic update(ref logic [3:0] value); value = value + 1; endtask\n"
        "  logic [3:0] actual;\n"
        "  initial update(actual);\n"
        "endmodule\n");
    VerilogConstantEnvironment environment{{"actual", value(3, 4)}};
    const auto result = execute_verilog_task_call(
        fixture.parsed.syntax, fixture.semantic, fixture.types,
        fixture.call("update"), environment);
    require(!result.completed &&
                std::any_of(result.diagnostics.begin(), result.diagnostics.end(),
                            [](const auto& diagnostic) {
                                return diagnostic.code == "UnsupportedCallableRefArgument";
                            }) &&
                known_value(environment, "actual") == 3,
            "ref argument execution must reject instead of copying values in and out");
}

} // namespace

int main() {
    try {
        test_sequential_flow_and_bounded_loops();
        test_casez_and_casex_item_wildcards();
        test_nested_calls_formal_propagation_and_function_expressions();
        test_known_width_conversion_on_input_and_writeback();
        test_declared_for_variable_has_loop_lifetime();
        test_output_formal_starts_uninitialized();
        test_unsupported_procedural_statements_are_diagnosed();
        test_recursion_call_depth_and_step_limits_are_transactional();
        test_named_binding_order_and_task_local_isolation();
        test_selected_and_concatenated_writeback_and_inout();
        test_overlapping_output_writeback_is_diagnosed_transactionally();
        test_ref_execution_does_not_fall_back_to_copy_semantics();
    } catch (const std::exception& error) {
        std::cerr << "EDA Verilog task execution test failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
