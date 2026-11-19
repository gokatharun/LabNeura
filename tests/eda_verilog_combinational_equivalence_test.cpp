#include "labneura/eda/verilog_native_lowering.h"
#include "support/eda_netlist_evaluator.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
using namespace labneura::eda;
using namespace labneura::eda::test;

constexpr std::size_t kDefaultMaxInputBits = 12;
constexpr std::size_t kHardMaxInputBits = 16;

void require(bool condition, std::string_view message) {
    if (!condition)
        throw std::runtime_error(std::string(message));
}

std::size_t exhaustive_input_bit_limit() {
    const char* configured = std::getenv("LABNEURA_NATIVE_EQUIV_MAX_INPUT_BITS");
    if (!configured || *configured == '\0')
        return kDefaultMaxInputBits;
    std::size_t consumed = 0;
    const std::string text(configured);
    const auto limit = std::stoull(text, &consumed);
    require(consumed == text.size() && limit <= kHardMaxInputBits,
            "LABNEURA_NATIVE_EQUIV_MAX_INPUT_BITS must be in [0, 16]");
    return static_cast<std::size_t>(limit);
}

std::string read_fixture(std::string_view name) {
    const auto path = std::filesystem::path(LABNEURA_NATIVE_COMB_FIXTURE_DIR) /
                      std::string(name);
    std::ifstream input(path, std::ios::binary);
    require(static_cast<bool>(input), "could not read combinational fixture " +
                                      path.string());
    return std::string(std::istreambuf_iterator<char>(input),
                       std::istreambuf_iterator<char>());
}

struct NativeDesign {
    VerilogSyntaxParseResult parsed;
    VerilogElaborationResult elaboration;
    VerilogTypeAnalysisResult types;
    VerilogSynthesisEligibilityResult eligibility;
    VerilogNativeLoweringResult lowered;

    NativeDesign(const std::string& source, std::string_view name)
        : parsed(parse_verilog_syntax(source)),
          elaboration(elaborate_verilog_design(
              parsed.syntax, VerilogElaborationOptions{.top_module = "m"})),
          types(analyze_verilog_expression_types(parsed.syntax,
                                                 elaboration.semantic)),
          eligibility(analyze_verilog_synthesis_eligibility(
              parsed.syntax, elaboration.semantic, types)),
          lowered(lower_verilog_native(parsed.syntax, elaboration, types,
                                       eligibility, "m")) {
        require(!parsed.has_errors(), "native fixture failed to parse");
        require(!elaboration.has_errors(), "native fixture failed to elaborate");
        if (!lowered.succeeded()) {
            std::string message = std::string(name) + " native fixture did not lower";
            for (const auto& diagnostic : lowered.diagnostics)
                message += "\n" + diagnostic.code + ": " + diagnostic.message;
            throw std::runtime_error(message);
        }
    }
};

struct SlangDesign {
    std::optional<Module> mapped;
    std::filesystem::path source_path;
    std::optional<std::string> unavailable_reason;

    explicit SlangDesign(const std::string& source) {
        const auto unique = std::chrono::steady_clock::now()
                                .time_since_epoch().count();
        source_path = std::filesystem::temp_directory_path() /
                      ("labneura_native_comb_equiv_" +
                       std::to_string(unique) + ".sv");
        {
            std::ofstream output(source_path);
            require(static_cast<bool>(output),
                    "could not create Slang reference source");
            output << source;
        }
        try {
            mapped.emplace(generic_map(parse_verilog(source_path.string())));
        } catch (const std::exception& error) {
            std::filesystem::remove(source_path);
            const std::string message = error.what();
            if (message.find("has no supported generic mapping") !=
                std::string::npos) {
                unavailable_reason = message;
            } else {
                throw;
            }
        }
        std::filesystem::remove(source_path);
    }
};

std::uint64_t port_unsigned(const VerificationPortValues& values,
                            std::string_view name) {
    const auto found = values.find(std::string(name));
    require(found != values.end(), "expected input port is missing: " +
                                   std::string(name));
    require(found->second.size() <= 64, "fixture input exceeds 64 bits");
    std::uint64_t result = 0;
    for (std::size_t bit = 0; bit < found->second.size(); ++bit)
        result |= static_cast<std::uint64_t>(found->second[bit] & 1U) << bit;
    return result;
}

std::int64_t signed_value(std::uint64_t value, std::size_t width) {
    if (width == 0) return 0;
    const std::uint64_t sign = std::uint64_t{1} << (width - 1);
    const std::uint64_t mask = (std::uint64_t{1} << width) - 1;
    value &= mask;
    return (value & sign) != 0
               ? static_cast<std::int64_t>(value) -
                     static_cast<std::int64_t>(std::uint64_t{1} << width)
               : static_cast<std::int64_t>(value);
}

VerificationBits bits(std::uint64_t value, std::size_t width) {
    VerificationBits result(width);
    for (std::size_t bit = 0; bit < width; ++bit)
        result[bit] = static_cast<std::uint8_t>((value >> bit) & 1U);
    return result;
}

VerificationPortValues exhaustive_inputs(const Module& module,
                                         std::uint64_t vector) {
    VerificationPortValues result;
    std::size_t offset = 0;
    for (const auto& port : module.get_ports()) {
        if (port.get_direction() != PortDirection::Input)
            continue;
        require(offset + port.get_width() <= 64,
                "fixture input vector exceeds 64 bits");
        result.emplace(port.get_name(), bits(vector >> offset,
                                             port.get_width()));
        offset += port.get_width();
    }
    return result;
}

std::size_t input_bit_count(const Module& module) {
    std::size_t total = 0;
    for (const auto& port : module.get_ports())
        if (port.get_direction() == PortDirection::Input)
            total += port.get_width();
    return total;
}

std::string format_ports(const VerificationPortValues& ports) {
    std::string formatted;
    for (const auto& [port, value] : ports) {
        if (!formatted.empty()) formatted += ", ";
        formatted += port + "=";
        for (const auto bit : value)
            formatted += bit == 0 ? '0' : '1';
    }
    return formatted;
}

bool check_outputs(std::string_view design_name,
                   const VerificationPortValues& expected,
                   const VerificationPortValues& actual,
                   const VerificationPortValues& inputs,
                   std::uint64_t vector, std::string_view implementation,
                   VerificationEvidence& evidence) {
    bool matches = true;
    for (const auto& [name, expected_bits] : expected) {
        const auto found = actual.find(name);
        if (found == actual.end() || found->second != expected_bits) {
            matches = false;
            if (!evidence.counterexample) {
                evidence.outcome = VerificationOutcome::Disagreed;
                evidence.counterexample = VerificationCounterexample{
                    static_cast<std::size_t>(vector), inputs, expected, actual};
                evidence.diagnostics.push_back(
                    {"OutputMismatch", std::string(design_name) + " " +
                         std::string(implementation) + " differs at output " + name,
                     name, std::nullopt});
            }
        }
    }
    for (const auto& [name, value] : actual) {
        (void)value;
        if (!expected.contains(name)) {
            matches = false;
            if (!evidence.counterexample) {
                evidence.outcome = VerificationOutcome::Disagreed;
                evidence.counterexample = VerificationCounterexample{
                    static_cast<std::size_t>(vector), inputs, expected, actual};
                evidence.diagnostics.push_back(
                    {"UnexpectedOutput", "unexpected observable output port '" +
                                              name + "'",
                     name, std::nullopt});
            }
        }
    }
    return matches;
}

std::string counterexample_message(std::string_view design_name,
                                  std::string_view implementation,
                                  const VerificationEvidence& evidence) {
    if (!evidence.counterexample)
        return std::string(design_name) + " " + std::string(implementation) +
               " differed without a counterexample record";
    const auto& counterexample = *evidence.counterexample;
    const std::string diagnostic = evidence.diagnostics.empty()
                                       ? std::string{}
                                       : "; " + evidence.diagnostics.front().message;
    return std::string(design_name) + " " + std::string(implementation) +
           " outcome=Disagreed, vector=" +
           std::to_string(counterexample.vector_or_frame) +
           ", inputs={" + format_ports(counterexample.inputs) +
           "}, expected={" + format_ports(counterexample.expected_outputs) +
           "}, actual={" + format_ports(counterexample.actual_outputs) + "}" +
           diagnostic;
}

using ExpectedOutputs = std::function<VerificationPortValues(
    const VerificationPortValues&)>;

void exhaustively_compare(std::string_view name, std::string_view fixture_name,
                          const ExpectedOutputs& expected_outputs) {
    const std::string source = read_fixture(fixture_name);
    NativeDesign native(source, name);
    SlangDesign slang(source);
    const Module& native_module = *native.lowered.module;
    const Module* slang_module = slang.mapped ? &*slang.mapped : nullptr;
    const auto input_bits = input_bit_count(native_module);
    const auto limit = exhaustive_input_bit_limit();
    require(input_bits <= limit,
            std::string(name) + " requires " + std::to_string(input_bits) +
                " exhaustive input bits, exceeding configured cap " +
                std::to_string(limit));
    require(input_bits < 63, "input space is too large to enumerate safely");
    const std::uint64_t vector_count = std::uint64_t{1} << input_bits;
    VerificationEvidence native_evidence;
    native_evidence.kind = VerificationKind::ExhaustiveCombinational;
    native_evidence.outcome = VerificationOutcome::Agreed;
    VerificationEvidence differential_evidence;
    differential_evidence.kind = VerificationKind::ExhaustiveCombinational;
    differential_evidence.outcome = VerificationOutcome::Agreed;
    bool slang_evaluation_unavailable = false;
    if (slang.unavailable_reason) {
        slang_evaluation_unavailable = true;
        differential_evidence.outcome = VerificationOutcome::Unavailable;
        differential_evidence.diagnostics.push_back(
            {"SlangGenericMappingUnsupported", *slang.unavailable_reason,
             std::nullopt, std::nullopt});
    }

    for (std::uint64_t vector = 0; vector < vector_count; ++vector) {
        const auto inputs = exhaustive_inputs(native_module, vector);
        const auto expected = expected_outputs(inputs);
        const auto native_eval = evaluate_combinational_netlist(native_module,
                                                                 inputs);
        require(native_eval.succeeded(),
                std::string(name) + " native netlist evaluation failed: " +
                    (native_eval.diagnostics.empty()
                         ? "no diagnostic"
                         : native_eval.diagnostics.front().message));
        native_evidence.checked_vectors = static_cast<std::size_t>(vector + 1);
        check_outputs(name, expected, native_eval.outputs, inputs, vector,
                      "native", native_evidence);

        if (!slang_evaluation_unavailable) {
            const auto slang_eval = evaluate_combinational_netlist(*slang_module,
                                                                    inputs);
            if (slang_eval.status == NetlistVerificationStatus::UnsupportedCell) {
                slang_evaluation_unavailable = true;
                if (differential_evidence.outcome != VerificationOutcome::Disagreed)
                    differential_evidence.outcome = VerificationOutcome::Unavailable;
                if (slang_eval.diagnostics.empty()) {
                    differential_evidence.diagnostics.push_back(
                        {"UnsupportedSlangMappedCell",
                         "Slang-mapped netlist uses a cell outside the bounded evaluator subset",
                         std::nullopt, std::nullopt});
                } else {
                    differential_evidence.diagnostics = slang_eval.diagnostics;
                }
                continue;
            }
            require(slang_eval.succeeded(),
                    std::string(name) + " Slang-mapped netlist evaluation failed: " +
                        (slang_eval.diagnostics.empty()
                             ? "no diagnostic"
                             : slang_eval.diagnostics.front().message));
            differential_evidence.checked_vectors =
                static_cast<std::size_t>(vector + 1);
            check_outputs(name, expected, slang_eval.outputs, inputs, vector,
                          "Slang-mapped", differential_evidence);
            check_outputs(name, native_eval.outputs, slang_eval.outputs, inputs,
                          vector, "native-versus-Slang", differential_evidence);
        }
    }
    require(native_evidence.kind == VerificationKind::ExhaustiveCombinational &&
                native_evidence.checked_vectors == vector_count,
            "native exhaustive verification evidence was not recorded correctly");
    if (native_evidence.outcome == VerificationOutcome::Disagreed) {
        std::cout << "FAIL: " << name
                  << " native outcome=Disagreed, kind=ExhaustiveCombinational, "
                  << native_evidence.checked_vectors << "/" << vector_count
                  << " vectors evaluated\n";
        throw std::runtime_error(counterexample_message(name, "native", native_evidence));
    }
    std::cout << "PASS: " << name
              << " native outcome=Agreed, kind=ExhaustiveCombinational, "
              << native_evidence.checked_vectors << " two-state input vectors ("
              << input_bits << " input bits)\n";
    if (differential_evidence.outcome == VerificationOutcome::Disagreed)
        throw std::runtime_error(
            counterexample_message(name, "native-vs-Slang", differential_evidence));
    if (slang_evaluation_unavailable) {
        std::cout << "UNAVAILABLE: " << name
                  << " Slang-mapped differential after "
                  << differential_evidence.checked_vectors
                  << " vectors; outcome=Unavailable; "
                  << (differential_evidence.diagnostics.empty()
                          ? "unsupported cell diagnostic missing"
                          : differential_evidence.diagnostics.front().message)
                  << '\n';
    } else {
        require(differential_evidence.kind ==
                    VerificationKind::ExhaustiveCombinational &&
                    differential_evidence.outcome == VerificationOutcome::Agreed &&
                    differential_evidence.checked_vectors == vector_count,
                "native-vs-Slang exhaustive evidence was not complete");
        std::cout << "PASS: " << name
                  << " Slang differential outcome=Agreed, "
                  << differential_evidence.checked_vectors << " vectors\n";
    }
}

void test_gates_and_adders() {
    exhaustively_compare(
        "gates_and_adders", "gates_and_adders.sv",
        [](const VerificationPortValues& inputs) {
            const auto a = port_unsigned(inputs, "a") != 0;
            const auto b = port_unsigned(inputs, "b") != 0;
            const auto cin = port_unsigned(inputs, "cin") != 0;
            return VerificationPortValues{
                {"and_result", bits(a && b, 1)},
                {"or_result", bits(a || b, 1)},
                {"xor_result", bits(a != b, 1)},
                {"half_sum", bits(a != b, 1)},
                {"half_carry", bits(a && b, 1)},
                {"full_sum", bits(a ^ b ^ cin, 1)},
                {"full_carry", bits((a && b) || (a && cin) || (b && cin), 1)}};
        });
}

void test_arithmetic_width_and_truncation() {
    exhaustively_compare(
        "arithmetic", "arithmetic.sv",
        [](const VerificationPortValues& inputs) {
            const auto a = port_unsigned(inputs, "a");
            const auto b = port_unsigned(inputs, "b");
            return VerificationPortValues{
                {"sum", bits((a + b) & 7U, 3)},
                {"difference", bits((a - b) & 7U, 3)}};
        });
}

void test_comparison_signedness_and_logical_bitwise() {
    exhaustively_compare(
        "comparison_and_logic", "comparison_and_logic.sv",
        [](const VerificationPortValues& inputs) {
            const auto a = port_unsigned(inputs, "a");
            const auto b = port_unsigned(inputs, "b");
            const auto ua = port_unsigned(inputs, "unsigned_a");
            const auto ub = port_unsigned(inputs, "unsigned_b");
            const auto sa = port_unsigned(inputs, "signed_a");
            const auto sb = port_unsigned(inputs, "signed_b");
            return VerificationPortValues{
                {"bitwise_and", bits(a & b, 2)},
                {"logical_and", bits(a != 0 && b != 0, 1)},
                {"equal", bits(a == b, 1)},
                {"unsigned_less", bits(ua < ub, 1)},
                {"signed_less", bits(signed_value(sa, 2) < signed_value(sb, 2), 1)}};
        });
}

void test_shifts_and_signed_conversion() {
    exhaustively_compare(
        "shifts_and_conversions", "shifts_and_conversions.sv",
        [](const VerificationPortValues& inputs) {
            const auto value = port_unsigned(inputs, "value");
            const auto signed_input = port_unsigned(inputs, "signed_value");
            const auto unsigned_input = port_unsigned(inputs, "unsigned_value");
            return VerificationPortValues{
                {"left_shifted", bits((value << 1) & 0xfU, 4)},
                {"sign_extended", bits(static_cast<std::uint64_t>(
                                            signed_value(signed_input, 3)) & 0x1fU,
                                        5)},
                {"zero_extended", bits(unsigned_input, 5)}};
        });
}

void test_packed_ranges_selections_and_concatenations() {
    exhaustively_compare(
        "ranges_selects_and_concat", "ranges_selects_and_concat.sv",
        [](const VerificationPortValues& inputs) {
            const auto value = port_unsigned(inputs, "descending");
            const auto bit0 = value & 1U;
            const auto selected = (value >> 1U) & 7U;
            const auto concatenated = (bit0 << 4U) | (selected << 1U) | bit0;
            const auto replicated = ((value & 3U) << 2U) | (value & 3U);
            return VerificationPortValues{
                {"ascending", bits(value, 4)},
                {"selected", bits(selected, 3)},
                {"concatenated", bits(concatenated, 5)},
                {"replicated", bits(replicated, 4)}};
        });
}

void test_conditional_case_and_blocking_order() {
    exhaustively_compare(
        "conditional_and_case", "conditional_and_case.sv",
        [](const VerificationPortValues& inputs) {
            const auto a = port_unsigned(inputs, "a");
            const auto b = port_unsigned(inputs, "b");
            const auto sel = port_unsigned(inputs, "sel") != 0;
            const auto choice = port_unsigned(inputs, "choice");
            const auto q = sel ? (a & b) : a;
            std::uint64_t selected = 0;
            switch (choice) {
                case 0: selected = a; break;
                case 1: selected = b; break;
                case 2: selected = a ^ b; break;
                default: selected = a | b; break;
            }
            return VerificationPortValues{{"q", bits(q, 2)},
                                          {"selected", bits(selected, 2)}};
        });
}

void test_partial_bit_assignments_and_parameters() {
    exhaustively_compare(
        "partial_bits_and_parameters", "partial_bits_and_parameters.sv",
        [](const VerificationPortValues& inputs) {
            const auto a = port_unsigned(inputs, "a");
            const auto b = port_unsigned(inputs, "b");
            const auto sel = port_unsigned(inputs, "sel") != 0;
            auto partial = ((a & 3U) << 2U) | (b & 3U);
            const auto bit0 = sel ? (a & 1U) : ((b >> 1U) & 1U);
            partial = (partial & ~std::uint64_t{1}) | bit0;
            return VerificationPortValues{
                {"sum", bits((a + b) & 7U, 3)},
                {"partial", bits(partial, 4)}};
        });
}

} // namespace

int main() {
    std::vector<std::string> failures;
    const auto run = [&](const auto& test) {
        try {
            test();
        } catch (const std::exception& error) {
            failures.push_back(error.what());
        }
    };
    run(test_gates_and_adders);
    run(test_arithmetic_width_and_truncation);
    run(test_comparison_signedness_and_logical_bitwise);
    run(test_shifts_and_signed_conversion);
    run(test_packed_ranges_selections_and_concatenations);
    run(test_conditional_case_and_blocking_order);
    run(test_partial_bit_assignments_and_parameters);
    if (!failures.empty()) {
        std::cerr << "EDA native combinational equivalence test failed:\n";
        for (const auto& failure : failures)
            std::cerr << " - " << failure << '\n';
        return 1;
    }
    return 0;
}
