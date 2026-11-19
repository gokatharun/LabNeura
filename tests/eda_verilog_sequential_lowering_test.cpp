#include "labneura/eda/verilog_native_lowering.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
using namespace labneura::eda;

void require(bool condition, std::string_view message) {
    if (!condition) throw std::runtime_error(std::string(message));
}

struct Fixture {
    VerilogSyntaxParseResult parsed;
    VerilogElaborationResult elaboration;
    VerilogTypeAnalysisResult types;
    VerilogSynthesisEligibilityResult eligibility;
    VerilogNativeLoweringResult lowered;

    explicit Fixture(std::string source)
        : parsed(parse_verilog_syntax(std::move(source))),
          elaboration(elaborate_verilog_design(
              parsed.syntax, VerilogElaborationOptions{.top_module = "m"})),
          types(analyze_verilog_expression_types(parsed.syntax,
                                                 elaboration.semantic)),
          eligibility(analyze_verilog_synthesis_eligibility(
              parsed.syntax, elaboration.semantic, types)),
          lowered(lower_verilog_native(parsed.syntax, elaboration, types,
                                       eligibility, "m")) {
        require(!parsed.has_errors(), "sequential fixture did not parse");
        require(!elaboration.has_errors(), "sequential fixture did not elaborate");
    }
};

const Instance* find_cell(const Module& module, std::string_view type) {
    const auto& instances = module.get_instances();
    const auto found = std::find_if(instances.begin(), instances.end(),
        [&](const auto& instance) { return instance.get_cell_type() == type; });
    return found == instances.end() ? nullptr : &*found;
}

std::string pin(const Instance& instance, std::string_view name) {
    const auto& pins = instance.get_connections();
    const auto found = std::find_if(pins.begin(), pins.end(),
        [&](const auto& connection) { return connection.get_pin() == name; });
    return found == pins.end() ? std::string{} : found->get_net();
}

bool diagnostic(const VerilogNativeLoweringResult& result,
                std::string_view code) {
    return std::any_of(result.diagnostics.begin(), result.diagnostics.end(),
        [&](const auto& entry) { return entry.code == code; });
}

void test_plain_dff_and_preedge_rhs() {
    Fixture f("module m(input logic clk, d, output logic q);\n"
              "always_ff @(posedge clk) q <= d;\nendmodule\n");
    require(f.lowered.succeeded(), "plain positive-edge DFF did not lower");
    const auto* dff = find_cell(*f.lowered.module, "DFF");
    require(dff && pin(*dff, "D") == "d" && pin(*dff, "CLK") == "clk",
            std::string("DFF pins were D=") + (dff ? pin(*dff, "D") : "<missing>") +
                ", CLK=" + (dff ? pin(*dff, "CLK") : "<missing>"));
    require(!pin(*dff, "Y").empty(), "DFF output pin was not connected");
}

void test_negedge_clock_is_inverted() {
    Fixture f("module m(input logic clk, d, output logic q);\n"
              "always @(negedge clk) q <= d;\nendmodule\n");
    require(f.lowered.succeeded(), "negative-edge DFF did not lower");
    const auto* inverter = find_cell(*f.lowered.module, "INV");
    const auto* dff = find_cell(*f.lowered.module, "DFF");
    require(inverter && pin(*inverter, "A") == "clk" && dff &&
                pin(*dff, "CLK") == pin(*inverter, "Y"),
            "negative edge was not represented with an inverted clock");
}

void test_enable_and_source_order_priority() {
    Fixture f("module m(input logic clk, en, a, b, output logic q);\n"
              "always_ff @(posedge clk) begin\n"
              "  if (en) q <= a;\n"
              "  if (en) q <= b;\n"
              "end\nendmodule\n");
    require(f.lowered.succeeded(), "conditional sequential writes did not lower");
    const auto& module = *f.lowered.module;
    const auto* latest_assignment = find_cell(module, "MUX2");
    const auto* flop = find_cell(module, "DFF");
    require(latest_assignment && flop,
            "conditional write did not create hold muxing and a DFF");
    const auto muxes = [&] {
        std::vector<const Instance*> result;
        for (const auto& instance : module.get_instances())
            if (instance.get_cell_type() == "MUX2") result.push_back(&instance);
        return result;
    }();
    require(flop && std::any_of(muxes.begin(), muxes.end(), [&](const auto* mux) {
                return pin(*mux, "A1") == "b" && pin(*mux, "S") == "en" &&
                       pin(*mux, "Y") == pin(*flop, "D");
            }),
            "later source-order NBA did not have priority at the DFF input");
}

void test_sync_and_async_reset_cells() {
    Fixture sync("module m(input logic clk, rst, d, output logic q);\n"
                 "always_ff @(posedge clk) if (rst) q <= 1'b0; else q <= d;\n"
                 "endmodule\n");
    require(sync.lowered.succeeded() && find_cell(*sync.lowered.module, "DFF") &&
                find_cell(*sync.lowered.module, "MUX2"),
            "synchronous reset was not lowered as a data mux");

    Fixture async("module m(input logic clk, rst_n, d, output logic q);\n"
                  "always_ff @(posedge clk or negedge rst_n)\n"
                  "  if (!rst_n) q <= 1'b0; else q <= d;\nendmodule\n");
    require(async.lowered.succeeded(), "active-low asynchronous reset did not lower");
    const auto* reset_flop = find_cell(*async.lowered.module, "DFFR");
    require(reset_flop && pin(*reset_flop, "RESET_B") == "rst_n",
            "active-low reset did not use the existing DFFR reset pin");
}

void test_reset_to_one_and_unsupported_reset() {
    Fixture set("module m(input logic clk, rst, d, output logic q);\n"
                "always_ff @(posedge clk or posedge rst)\n"
                "  if (rst) q <= 1'b1; else q <= d;\nendmodule\n");
    require(set.lowered.succeeded() && find_cell(*set.lowered.module, "DFFS"),
            "constant-one asynchronous reset did not use DFFS");

    Fixture invalid("module m(input logic clk, rst, d, output logic q);\n"
                    "always_ff @(posedge clk or posedge rst)\n"
                    "  if (rst) q <= d; else q <= d;\nendmodule\n");
    require(!invalid.lowered.succeeded() && !invalid.lowered.module &&
                (!invalid.lowered.diagnostics.empty() ||
                 diagnostic(invalid.lowered, "UnsupportedAsynchronousReset")),
            "unproven asynchronous reset was not rejected with a diagnostic");
}
} // namespace

int main() {
    try {
        test_plain_dff_and_preedge_rhs();
        test_negedge_clock_is_inverted();
        test_enable_and_source_order_priority();
        test_sync_and_async_reset_cells();
        test_reset_to_one_and_unsupported_reset();
    } catch (const std::exception& error) {
        std::cerr << "EDA Verilog sequential lowering test failed: "
                  << error.what() << '\n';
        return 1;
    }
    return 0;
}
