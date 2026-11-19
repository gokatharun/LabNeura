#include "labneura/eda/verilog_native_lowering.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#ifndef LABNEURA_NATIVE_FORMAL_FIXTURE_DIR
#error "LABNEURA_NATIVE_FORMAL_FIXTURE_DIR must point at formal SV fixtures"
#endif

namespace {
using namespace labneura::eda;

void require(bool condition, std::string_view message) {
    if (!condition)
        throw std::runtime_error(std::string(message));
}

std::string read_text(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input)
        throw std::runtime_error("could not read fixture: " + path.string());
    std::ostringstream contents;
    contents << input.rdbuf();
    return contents.str();
}

std::string shell_quote(const std::string& value) {
    std::string result = "'";
    for (const char character : value) {
        if (character == '\'')
            result += "'\\''";
        else
            result += character;
    }
    result += "'";
    return result;
}

std::string yosys_quote(const std::string& value) {
    std::string result = "\"";
    for (const char character : value) {
        if (character == '\\' || character == '\"')
            result += '\\';
        result += character;
    }
    result += "\"";
    return result;
}

std::string cell_alias(std::string_view type) {
    constexpr char hex[] = "0123456789abcdef";
    std::string result = "__labneura_cell__";
    for (const unsigned char character : type) {
        result.push_back(hex[character >> 4]);
        result.push_back(hex[character & 0x0f]);
    }
    return result;
}

std::string cell_model(const std::string& type, bool break_and = false) {
    const auto alias = cell_alias(type);
    if (type == "AND2")
        return "module " + alias + " (A, B, Y);\n"
               "  input wire A, B; output wire Y;\n  assign Y = A " +
               std::string(break_and ? "|" : "&") + " B;\nendmodule\n";
    if (type == "OR2")
        return "module " + alias + " (A, B, Y);\n"
               "  input wire A, B; output wire Y;\n  assign Y = A | B;\nendmodule\n";
    if (type == "XOR2")
        return "module " + alias + " (A, B, Y);\n"
               "  input wire A, B; output wire Y;\n  assign Y = A ^ B;\nendmodule\n";
    if (type == "INV")
        return "module " + alias + " (A, Y);\n"
               "  input wire A; output wire Y;\n  assign Y = ~A;\nendmodule\n";
    if (type == "MUX2")
        return "module " + alias + " (A0, A1, S, Y);\n"
               "  input wire A0, A1, S; output wire Y;\n"
               "  assign Y = S ? A1 : A0;\nendmodule\n";
    if (type == "BUF")
        return "module " + alias + " (A, Y);\n"
               "  input wire A; output wire Y;\n  assign Y = A;\nendmodule\n";
    if (type == "TIEHI")
        return "module " + alias + " (Y);\n"
               "  output wire Y; assign Y = 1'b1;\nendmodule\n";
    if (type == "TIELO")
        return "module " + alias + " (Y);\n"
               "  output wire Y; assign Y = 1'b0;\nendmodule\n";
    throw std::runtime_error("formal harness has no model for cell " + type);
}

void replace_blackbox(std::string& verilog, const std::string& type,
                      bool break_and = false) {
    const auto alias = cell_alias(type);
    const auto marker = "(* blackbox *) module " + alias;
    const auto begin = verilog.find(marker);
    require(begin != std::string::npos,
            "serialized native design omitted cell blackbox " + type);
    const auto end_marker = verilog.find("endmodule", begin);
    require(end_marker != std::string::npos,
            "serialized native cell declaration is malformed for " + type);
    const auto end = end_marker + std::string_view("endmodule").size();
    verilog.replace(begin, end - begin, cell_model(type, break_and));
}

struct LoweredFixture {
    Module module;
    std::string source;
};

LoweredFixture lower_fixture(const std::filesystem::path& fixture_path) {
    const auto source = read_text(fixture_path);
    auto parsed = parse_verilog_syntax(source);
    require(!parsed.has_errors(), "formal fixture failed native syntax parsing: " +
                                      fixture_path.filename().string());
    auto elaboration = elaborate_verilog_design(
        parsed.syntax, VerilogElaborationOptions{.top_module = "m"});
    require(!elaboration.has_errors(), "formal fixture failed native elaboration: " +
                                           fixture_path.filename().string());
    auto types = analyze_verilog_expression_types(parsed.syntax,
                                                   elaboration.semantic);
    auto eligibility = analyze_verilog_synthesis_eligibility(
        parsed.syntax, elaboration.semantic, types);
    auto lowered = lower_verilog_native(parsed.syntax, elaboration, types,
                                       eligibility, "m");
    if (!lowered.succeeded()) {
        std::string detail;
        for (const auto& diagnostic : lowered.diagnostics)
            detail += "\n  " + diagnostic.code + ": " + diagnostic.message;
        throw std::runtime_error("formal fixture failed native lowering: " +
                                 fixture_path.filename().string() + detail);
    }
    return {std::move(*lowered.module), source};
}

std::string renamed_gold(std::string source) {
    const auto module = source.find("module m");
    require(module != std::string::npos,
            "formal source fixture must define module m");
    source.replace(module, std::string_view("module m").size(), "module gold");
    return source;
}

struct YosysRun {
    int status = -1;
    std::string output;
};

YosysRun run_proof(const Module& candidate, const std::string& gold_source,
                   const std::filesystem::path& directory,
                   std::string_view label, bool break_and = false) {
    std::filesystem::create_directories(directory);
    auto verilog = to_verilog(candidate);
    std::set<std::string> cell_types;
    for (const auto& instance : candidate.get_instances())
        cell_types.insert(instance.get_cell_type());
    for (const auto& type : cell_types)
        replace_blackbox(verilog, type, break_and && type == "AND2");
    verilog += "\n" + renamed_gold(gold_source) + "\n";

    const auto netlist_path = directory / (std::string(label) + ".sv");
    const auto script_path = directory / (std::string(label) + ".ys");
    const auto log_path = directory / (std::string(label) + ".log");
    {
        std::ofstream output(netlist_path);
        require(static_cast<bool>(output), "could not write isolated Yosys input");
        output << verilog;
    }
    {
        std::ofstream script(script_path);
        require(static_cast<bool>(script), "could not write isolated Yosys script");
        script << "read_verilog -formal -sv " << yosys_quote(netlist_path.string())
               << "\nproc\nequiv_make gold m equiv\nprep -flatten -top equiv\n"
               << "equiv_simple\nequiv_status -assert\n";
    }

    const std::string command = "yosys -Q -T -s " + shell_quote(script_path.string()) +
                                " > " + shell_quote(log_path.string()) + " 2>&1";
    YosysRun result;
    result.status = std::system(command.c_str());
    std::ifstream log(log_path);
    std::ostringstream contents;
    contents << log.rdbuf();
    result.output = contents.str();
    return result;
}

void prove_fixture(const std::filesystem::path& fixture_path,
                   const std::filesystem::path& directory) {
    auto lowered = lower_fixture(fixture_path);
    const auto result = run_proof(lowered.module, lowered.source, directory,
                                  fixture_path.stem().string());
    if (result.status != 0) {
        throw std::runtime_error("Yosys did not prove " +
                                 fixture_path.filename().string() +
                                 " equivalent to native lowering:\n" + result.output);
    }
    std::cout << "PROVEN: " << fixture_path.filename().string() << '\n';
}

void test_incorrect_native_candidate_fails_proof(
    const std::filesystem::path& fixture_path,
    const std::filesystem::path& directory) {
    auto lowered = lower_fixture(fixture_path);
    const auto result = run_proof(lowered.module, lowered.source, directory,
                                  "deliberately_wrong_and_model", true);
    require(result.status != 0,
            "Yosys incorrectly accepted a deliberately wrong AND2 candidate model");
    const auto unproven = result.output.find("unproven $equiv cells");
    const auto assertion = result.output.find("equiv_status -assert");
    require(unproven != std::string::npos && assertion != std::string::npos,
            "Yosys failed for a reason other than a detected inequivalence:\n" +
                result.output);
    std::cout << "DETECTED: deliberately inequivalent AND2 model\n";
}

} // namespace

int main() {
    if (std::system("command -v yosys >/dev/null 2>&1") != 0) {
        std::cout << "UNAVAILABLE: Yosys formal equivalence test skipped\n";
        return 77;
    }

    try {
        const auto unique = std::chrono::steady_clock::now()
                                .time_since_epoch().count();
        const auto directory = std::filesystem::temp_directory_path() /
            ("labneura_native_formal_" + std::to_string(unique));
        const std::filesystem::path fixtures(LABNEURA_NATIVE_FORMAL_FIXTURE_DIR);
        prove_fixture(fixtures / "boolean_cells.sv", directory / "boolean_cells");
        prove_fixture(fixtures / "conditional_mux.sv", directory / "conditional_mux");
        prove_fixture(fixtures / "two_bit_add.sv", directory / "two_bit_add");
        test_incorrect_native_candidate_fails_proof(
            fixtures / "boolean_cells.sv", directory / "negative");
        std::filesystem::remove_all(directory);
    } catch (const std::exception& exception) {
        std::cerr << "EDA native formal equivalence test failed: "
                  << exception.what() << '\n';
        return 1;
    }
    return 0;
}
