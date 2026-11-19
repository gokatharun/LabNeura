#include "labneura/eda/verilog_native_lowering.h"
#include "support/eda_netlist_evaluator.h"

#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
using namespace labneura::eda;
using namespace labneura::eda::test;

constexpr std::uint64_t kSeed = 0x4d3731375eedULL;
constexpr std::size_t kGeneratedExpressions = 24;
constexpr std::size_t kInputBits = 9; // a[3:0], b[3:0], s
constexpr std::size_t kExhaustiveVectors = std::size_t{1} << kInputBits;

void require(bool condition, std::string_view message) {
    if (!condition)
        throw std::runtime_error(std::string(message));
}

struct Analysis {
    VerilogSyntaxParseResult parsed;
    VerilogElaborationResult elaboration;
    VerilogTypeAnalysisResult types;
    VerilogSynthesisEligibilityResult eligibility;

    explicit Analysis(std::string source)
        : parsed(parse_verilog_syntax(std::move(source))),
          elaboration(elaborate_verilog_design(
              parsed.syntax, VerilogElaborationOptions{.top_module = "m"})),
          types(analyze_verilog_expression_types(parsed.syntax,
                                                 elaboration.semantic)),
          eligibility(analyze_verilog_synthesis_eligibility(
              parsed.syntax, elaboration.semantic, types)) {}
};

struct Expr {
    enum class Kind { InputA, InputB, Select, Constant, Not, Binary, Shift, Mux };
    Kind kind = Kind::InputA;
    std::uint8_t constant = 0;
    char operation = '&';
    std::uint8_t shift = 0;
    std::shared_ptr<Expr> lhs;
    std::shared_ptr<Expr> rhs;
    std::shared_ptr<Expr> third;

    std::string source() const {
        switch (kind) {
        case Kind::InputA: return "a";
        case Kind::InputB: return "b";
        case Kind::Select: return "s";
        case Kind::Constant: {
            std::ostringstream stream;
            stream << "4'h" << std::hex << static_cast<unsigned>(constant);
            return stream.str();
        }
        case Kind::Not: return "(~(" + lhs->source() + "))";
        case Kind::Binary:
            return "((" + lhs->source() + ") " + operation + " (" +
                   rhs->source() + "))";
        case Kind::Shift:
            return "((" + lhs->source() + ") " +
                   std::string(2, operation) + " 2'd" +
                   std::to_string(shift) + ")";
        case Kind::Mux:
            return "(s ? (" + lhs->source() + ") : (" + rhs->source() + "))";
        }
        throw std::logic_error("invalid randomized expression kind");
    }

    std::uint8_t evaluate(std::uint8_t a, std::uint8_t b,
                          std::uint8_t select) const {
        constexpr std::uint8_t mask = 0x0f;
        switch (kind) {
        case Kind::InputA: return a & mask;
        case Kind::InputB: return b & mask;
        case Kind::Select: return select ? 1 : 0;
        case Kind::Constant: return constant & mask;
        case Kind::Not: return static_cast<std::uint8_t>(~lhs->evaluate(a, b, select)) & mask;
        case Kind::Binary: {
            const auto left = lhs->evaluate(a, b, select);
            const auto right = rhs->evaluate(a, b, select);
            if (operation == '&') return static_cast<std::uint8_t>(left & right);
            if (operation == '|') return static_cast<std::uint8_t>(left | right);
            if (operation == '^') return static_cast<std::uint8_t>(left ^ right);
            if (operation == '+') return static_cast<std::uint8_t>(left + right) & mask;
            if (operation == '-') return static_cast<std::uint8_t>(left - right) & mask;
            throw std::logic_error("invalid randomized binary operator");
        }
        case Kind::Shift: {
            const auto value = lhs->evaluate(a, b, select);
            return operation == '<'
                       ? static_cast<std::uint8_t>(value << shift) & mask
                       : static_cast<std::uint8_t>(value >> shift) & mask;
        }
        case Kind::Mux:
            return select ? lhs->evaluate(a, b, select)
                          : rhs->evaluate(a, b, select);
        }
        throw std::logic_error("invalid randomized expression kind");
    }
};

std::shared_ptr<Expr> make_leaf(std::mt19937_64& random) {
    auto expression = std::make_shared<Expr>();
    switch (random() % 3) {
    case 0: expression->kind = Expr::Kind::InputA; break;
    case 1: expression->kind = Expr::Kind::InputB; break;
    default: expression->kind = Expr::Kind::Constant;
             expression->constant = static_cast<std::uint8_t>(random() & 0x0fU);
             break;
    }
    return expression;
}

std::shared_ptr<Expr> make_expression(std::mt19937_64& random,
                                      std::size_t depth) {
    if (depth == 0 || random() % 7 == 0)
        return make_leaf(random);

    auto expression = std::make_shared<Expr>();
    switch (random() % 6) {
    case 0:
        expression->kind = Expr::Kind::Not;
        expression->lhs = make_expression(random, depth - 1);
        break;
    case 1:
    case 2:
    case 3:
        expression->kind = Expr::Kind::Binary;
        expression->operation = random() % 5 == 0 ? '+' :
                                random() % 4 == 0 ? '-' :
                                random() % 3 == 0 ? '&' :
                                random() % 2 == 0 ? '|' : '^';
        expression->lhs = make_expression(random, depth - 1);
        expression->rhs = make_expression(random, depth - 1);
        break;
    case 4:
        expression->kind = Expr::Kind::Shift;
        expression->operation = random() % 2 == 0 ? '<' : '>';
        expression->shift = static_cast<std::uint8_t>(random() % 4);
        expression->lhs = make_expression(random, depth - 1);
        break;
    default:
        expression->kind = Expr::Kind::Mux;
        expression->lhs = make_expression(random, depth - 1);
        expression->rhs = make_expression(random, depth - 1);
        break;
    }
    return expression;
}

std::string build_random_source(std::vector<std::shared_ptr<Expr>>& expressions,
                                std::vector<std::pair<std::shared_ptr<Expr>,
                                                     std::shared_ptr<Expr>>>&
                                    procedural_expressions) {
    std::mt19937_64 random(kSeed);
    for (std::size_t i = 0; i < kGeneratedExpressions; ++i)
        expressions.push_back(make_expression(random, 3));
    for (std::size_t i = 0; i < 4; ++i)
        procedural_expressions.emplace_back(make_expression(random, 2),
                                            make_expression(random, 2));

    std::ostringstream source;
    source << "module m(input logic [3:0] a, b, input logic s,\n";
    for (std::size_t i = 0; i < expressions.size(); ++i)
        source << "  output logic [3:0] y" << i << ",\n";
    for (std::size_t i = 0; i < procedural_expressions.size(); ++i)
        source << "  output logic [3:0] p" << i
               << (i + 1 == procedural_expressions.size() ? ");\n" : ",\n");
    for (std::size_t i = 0; i < expressions.size(); ++i)
        source << "  assign y" << i << " = " << expressions[i]->source() << ";\n";
    source << "  always_comb begin\n";
    for (std::size_t i = 0; i < procedural_expressions.size(); ++i) {
        source << "    if (s) p" << i << " = "
               << procedural_expressions[i].first->source()
               << "; else p" << i << " = "
               << procedural_expressions[i].second->source() << ";\n";
    }
    source << "  end\nendmodule\n";
    return source.str();
}

VerificationBits bits(std::uint8_t value, std::size_t width) {
    VerificationBits result(width);
    for (std::size_t bit = 0; bit < width; ++bit)
        result[bit] = static_cast<std::uint8_t>((value >> bit) & 1U);
    return result;
}

std::string format_values(const VerificationPortValues& values) {
    std::string output;
    for (const auto& [name, value] : values) {
        if (!output.empty()) output += ", ";
        output += name + "=0b";
        for (auto bit = value.rbegin(); bit != value.rend(); ++bit)
            output += *bit == 0 ? '0' : '1';
    }
    return output;
}

void test_deterministic_random_expression_trees() {
    std::vector<std::shared_ptr<Expr>> expressions;
    std::vector<std::pair<std::shared_ptr<Expr>, std::shared_ptr<Expr>>>
        procedural_expressions;
    const auto source = build_random_source(expressions, procedural_expressions);
    Analysis analysis(source);
    require(!analysis.parsed.has_errors(), "randomized source failed to parse");
    require(!analysis.elaboration.has_errors(),
            "randomized source failed to elaborate");
    const auto lowered = lower_verilog_native(
        analysis.parsed.syntax, analysis.elaboration, analysis.types,
        analysis.eligibility, "m");
    if (!lowered.succeeded()) {
        std::string message = "randomized supported expressions did not lower; seed=" +
                              std::to_string(kSeed) + "\n" + source;
        for (const auto& diagnostic : lowered.diagnostics)
            message += "\n" + diagnostic.code + " at byte " +
                       std::to_string(diagnostic.location.byte_offset) + ": " +
                       diagnostic.message + "\nnear: " +
                       source.substr(diagnostic.location.byte_offset > 32
                                         ? diagnostic.location.byte_offset - 32 : 0,
                                     96);
        throw std::runtime_error(message);
    }

    Analysis repeated_analysis(source);
    const auto repeated = lower_verilog_native(
        repeated_analysis.parsed.syntax, repeated_analysis.elaboration,
        repeated_analysis.types, repeated_analysis.eligibility, "m");
    require(repeated.succeeded() &&
                to_verilog(*lowered.module) == to_verilog(*repeated.module),
            "repeated lowering changed deterministic netlist output");

    VerificationEvidence evidence;
    evidence.kind = VerificationKind::ExhaustiveCombinational;
    evidence.outcome = VerificationOutcome::Agreed;
    evidence.random_seed = kSeed;
    evidence.checked_vectors = kExhaustiveVectors;

    for (std::size_t vector = 0; vector < kExhaustiveVectors; ++vector) {
        const auto a = static_cast<std::uint8_t>(vector & 0x0fU);
        const auto b = static_cast<std::uint8_t>((vector >> 4) & 0x0fU);
        const auto select = static_cast<std::uint8_t>((vector >> 8) & 1U);
        const VerificationPortValues inputs{
            {"a", bits(a, 4)}, {"b", bits(b, 4)}, {"s", bits(select, 1)}};
        const auto evaluated = evaluate_combinational_netlist(
            *lowered.module, inputs);
        require(evaluated.succeeded(),
                "netlist evaluator rejected randomized lowered module");

        VerificationPortValues expected;
        for (std::size_t index = 0; index < expressions.size(); ++index)
            expected.emplace("y" + std::to_string(index),
                             bits(expressions[index]->evaluate(a, b, select), 4));
        for (std::size_t index = 0; index < procedural_expressions.size(); ++index) {
            const auto& [when_true, when_false] = procedural_expressions[index];
            expected.emplace("p" + std::to_string(index),
                             bits((select ? when_true : when_false)->evaluate(
                                      a, b, select), 4));
        }
        if (evaluated.outputs != expected) {
            evidence.outcome = VerificationOutcome::Disagreed;
            evidence.counterexample = VerificationCounterexample{
                vector, inputs, expected, evaluated.outputs};
            throw std::runtime_error(
                "randomized lowering mismatch; seed=" + std::to_string(kSeed) +
                ", vector=" + std::to_string(vector) + ", inputs={" +
                format_values(inputs) + "}, expected={" + format_values(expected) +
                "}, actual={" + format_values(evaluated.outputs) + "}");
        }
    }
    require(evidence.outcome == VerificationOutcome::Agreed &&
                evidence.checked_vectors == 512,
            "randomized exhaustive evidence accounting is inconsistent");
}

void require_atomic_failure(std::string_view name, std::string source,
                            std::string_view diagnostic_code = {}) {
    Analysis analysis(std::move(source));
    const auto result = lower_verilog_native(
        analysis.parsed.syntax, analysis.elaboration, analysis.types,
        analysis.eligibility, "m");
    require(!result.succeeded() && !result.module && !result.diagnostics.empty(),
            std::string(name) + " did not fail atomically with diagnostics");
    if (!diagnostic_code.empty()) {
        bool found = false;
        std::string actual;
        for (const auto& diagnostic : result.diagnostics)
        {
            if (!actual.empty()) actual += ", ";
            actual += diagnostic.code;
            found = found || diagnostic.code == diagnostic_code;
        }
        require(found, std::string(name) + " lacked expected diagnostic " +
                           std::string(diagnostic_code) + "; got: " + actual);
    }
}

void test_negative_subset_and_failure_atomicity() {
    require_atomic_failure("unsupported multiply",
        "module m(input logic [3:0] a, b, output logic [3:0] y); "
        "assign y = a * b; endmodule\n", "UnsupportedNativeBinaryOperator");
    require_atomic_failure("X constant",
        "module m(output logic y); assign y = 1'bx; endmodule\n",
        "FourStateConstantUnsupported");
    require_atomic_failure("dynamic select",
        "module m(input logic [3:0] a, input logic [1:0] i, output logic y); "
        "assign y = a[i]; endmodule\n", "DynamicNativeSelectUnsupported");
    require_atomic_failure("latch-producing process",
        "module m(input logic en, d, output logic q); "
        "always_latch if (en) q = d; endmodule\n");
    require_atomic_failure("multiple drivers",
        "module m(input logic a, b, output logic y); "
        "assign y = a; assign y = b; endmodule\n");
    require_atomic_failure("unresolved dependency",
        "module m(output logic y); assign y = missing; endmodule\n");
    require_atomic_failure("hierarchy",
        "module child(input logic a, output logic y); assign y = a; endmodule "
        "module m(input logic a, output logic y); child u(.a(a), .y(y)); endmodule\n",
        "UnsupportedNativeHierarchy");
    require_atomic_failure("inout port",
        "module m(inout wire io); endmodule\n");
    require_atomic_failure("unproven asynchronous reset",
        "module m(input logic clk, rst, d, output logic q); "
        "always_ff @(posedge clk or posedge rst) if (rst) q <= d; "
        "else q <= d; endmodule\n");
}

void test_mismatched_analysis_identity_is_rejected() {
    Analysis first("module m(input logic a, output logic y); "
                   "assign y = a; endmodule\n");
    Analysis second("module m(input logic b, output logic y); "
                    "assign y = b; endmodule\n");
    const auto result = lower_verilog_native(
        first.parsed.syntax, first.elaboration, first.types,
        second.eligibility, "m");
    require(!result.succeeded() && !result.module && !result.diagnostics.empty(),
            "mismatched semantic analysis was not rejected atomically");
}

} // namespace

int main() {
    try {
        test_deterministic_random_expression_trees();
        test_negative_subset_and_failure_atomicity();
        test_mismatched_analysis_identity_is_rejected();
    } catch (const std::exception& error) {
        std::cerr << "EDA randomized native lowering test failed: " << error.what()
                  << '\n';
        return 1;
    }
    return 0;
}
