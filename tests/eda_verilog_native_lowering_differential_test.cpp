#include "labneura/eda/netlist.h"
#include "labneura/eda/verilog_native_lowering.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {
using namespace labneura::eda;

void require(bool condition, std::string_view message) {
    if (!condition)
        throw std::runtime_error(std::string(message));
}

struct NativeFixture {
    VerilogSyntaxParseResult parsed;
    VerilogElaborationResult elaboration;
    VerilogTypeAnalysisResult types;
    VerilogSynthesisEligibilityResult eligibility;
    VerilogNativeLoweringResult lowered;

    explicit NativeFixture(const std::string& source)
        : parsed(parse_verilog_syntax(source)),
          elaboration(elaborate_verilog_design(
              parsed.syntax, VerilogElaborationOptions{.top_module = "m"})),
          types(analyze_verilog_expression_types(parsed.syntax,
                                                 elaboration.semantic)),
          eligibility(analyze_verilog_synthesis_eligibility(
              parsed.syntax, elaboration.semantic, types)),
          lowered(lower_verilog_native(parsed.syntax, elaboration, types,
                                       eligibility, "m")) {
        require(!parsed.has_errors() && !elaboration.has_errors(),
                "native differential fixture did not parse and elaborate");
        require(lowered.succeeded(), "native differential fixture did not lower");
    }
};

bool input_pin(const Instance& cell, std::string_view pin_name,
               const std::map<std::string, bool>& values, bool& result) {
    for (const auto& connection : cell.get_connections()) {
        if (connection.get_pin() == pin_name) {
            const auto found = values.find(connection.get_net());
            if (found == values.end()) return false;
            result = found->second;
            return true;
        }
    }
    throw std::runtime_error("test evaluator cannot find cell pin " +
                             std::string(pin_name));
}

std::string output_net(const Instance& cell) {
    for (const auto& connection : cell.get_connections())
        if (connection.get_pin() == "Y" || connection.get_pin() == "O")
            return connection.get_net();
    throw std::runtime_error("test evaluator cannot find a cell output");
}

bool eval_cell(const Instance& cell, const std::map<std::string, bool>& values,
               bool& known) {
    bool a = false;
    bool b = false;
    known = true;
    const auto& type = cell.get_cell_type();
    if (type == "TIEHI") return true;
    if (type == "TIELO") return false;
    if (type == "INV" || type == "NOT") {
        if (!input_pin(cell, "A", values, a)) { known = false; return false; }
        return !a;
    }
    if (type == "BUF") {
        if (!input_pin(cell, "A", values, a)) { known = false; return false; }
        return a;
    }
    if (type == "AND2" || type == "OR2" || type == "XOR2") {
        if (!input_pin(cell, "A", values, a) ||
            !input_pin(cell, "B", values, b)) { known = false; return false; }
        if (type == "AND2") return a && b;
        if (type == "OR2") return a || b;
        return a != b;
    }
    if (type == "MUX2") {
        bool select = false;
        if (!input_pin(cell, "A0", values, a) ||
            !input_pin(cell, "A1", values, b) ||
            !input_pin(cell, "S", values, select)) { known = false; return false; }
        return select ? b : a;
    }
    throw std::runtime_error("test evaluator encountered unsupported cell " + type);
}

bool evaluate_single_bit_output(const Module& module, bool a, bool b) {
    std::map<std::string, bool> values{{"a", a}, {"b", b}};
    for (size_t iteration = 0; iteration <= module.get_instances().size(); ++iteration) {
        bool progress = false;
        for (const auto& cell : module.get_instances()) {
            bool known = false;
            const bool value = eval_cell(cell, values, known);
            if (known && !values.contains(output_net(cell))) {
                values.emplace(output_net(cell), value);
                progress = true;
            }
        }
        if (!progress) break;
    }
    const auto found = values.find("y");
    require(found != values.end(), "test evaluator could not resolve output y");
    return found->second;
}

std::string shell_quote(const std::string& value) {
    std::string quoted = "'";
    for (const char character : value) {
        if (character == '\'')
            quoted += "'\\''";
        else
            quoted += character;
    }
    quoted += "'";
    return quoted;
}

void implement_blackbox(std::string& verilog, std::string_view cell_alias,
                        std::string_view implementation) {
    const std::string marker = "(* blackbox *) module " + std::string(cell_alias);
    const auto start = verilog.find(marker);
    require(start != std::string::npos,
            "serialized native netlist omitted an expected generic cell model");
    const auto end_marker = verilog.find("endmodule", start);
    require(end_marker != std::string::npos,
            "serialized generic cell blackbox has no endmodule");
    const auto end = end_marker + std::string_view("endmodule").size();
    verilog.replace(start, end - start, implementation);
}

void optional_yosys_equivalence_proof(const Module& native_module,
                                      const std::filesystem::path& directory) {
    if (std::system("command -v yosys >/dev/null 2>&1") != 0) {
        std::cout << "SKIP: optional Yosys native lowering equivalence proof "
                     "(yosys not found)\n";
        return;
    }

    auto verilog = to_verilog(native_module);
    implement_blackbox(
        verilog, "__labneura_cell__414e4432",
        "module __labneura_cell__414e4432 (A, B, Y);\n"
        "  input wire A, B;\n  output wire Y;\n"
        "  assign Y = A & B;\nendmodule");
    implement_blackbox(
        verilog, "__labneura_cell__425546",
        "module __labneura_cell__425546 (A, Y);\n"
        "  input wire A;\n  output wire Y;\n"
        "  assign Y = A;\nendmodule");
    verilog += "\nmodule gold(input wire a, b, output wire y);\n"
               "  assign y = a & b;\nendmodule\n";

    const auto input = directory / "native_and_equiv.sv";
    const auto log = directory / "yosys_equiv.log";
    {
        std::ofstream output(input);
        require(static_cast<bool>(output), "could not create Yosys proof input");
        output << verilog;
    }
    const std::string command =
        "yosys -Q -T -p \"read_verilog -sv " + input.string() +
        "; flatten; equiv_make gold m equiv; hierarchy -top equiv; equiv_simple; "
        "equiv_status -assert\" > " + shell_quote(log.string()) + " 2>&1";
    const int status = std::system(command.c_str());
    if (status != 0) {
        std::ifstream result(log);
        const std::string details((std::istreambuf_iterator<char>(result)),
                                  std::istreambuf_iterator<char>());
        throw std::runtime_error("Yosys failed to prove native AND equivalence:\n" +
                                 details);
    }
    std::cout << "PASS: optional Yosys native lowering equivalence proof\n";
}

void test_native_and_slang_lowering_truth_table() {
    const std::string source =
        "module m(input logic a, b, output logic y);\n"
        "  assign y = a & b;\n"
        "endmodule\n";
    NativeFixture native(source);

    const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto path = std::filesystem::temp_directory_path() /
        ("labneura_native_lowering_diff_" + std::to_string(unique) + ".sv");
    const auto directory = path.parent_path() /
        ("labneura_native_lowering_diff_" + std::to_string(unique));
    try {
        std::filesystem::create_directories(directory);
        {
            std::ofstream output(path);
            require(static_cast<bool>(output), "could not create Slang differential fixture");
            output << source;
        }
        const Module slang_mapped = generic_map(parse_verilog(path.string()));
        for (unsigned vector = 0; vector < 4; ++vector) {
            const bool a = (vector & 1U) != 0;
            const bool b = (vector & 2U) != 0;
            const bool reference = evaluate_single_bit_output(slang_mapped, a, b);
            const bool actual = evaluate_single_bit_output(*native.lowered.module, a, b);
            require(actual == reference,
                    "native and Slang-backed lowering differ for an AND input vector");
        }
        optional_yosys_equivalence_proof(*native.lowered.module, directory);
    } catch (...) {
        std::filesystem::remove(path);
        std::filesystem::remove_all(directory);
        throw;
    }
    std::filesystem::remove(path);
    std::filesystem::remove_all(directory);
}
} // namespace

int main() {
    try {
        test_native_and_slang_lowering_truth_table();
    } catch (const std::exception& error) {
        std::cerr << "EDA Verilog native lowering differential test failed: "
                  << error.what() << '\n';
        return 1;
    }
    return 0;
}
