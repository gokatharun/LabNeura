#include "support/eda_netlist_evaluator.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
using namespace labneura::eda;
using namespace labneura::eda::test;

void require(bool condition, std::string_view message) {
    if (!condition) throw std::runtime_error(std::string(message));
}

Module one_cell(std::string type, std::vector<std::string> input_names,
                std::vector<PinConnection> connections) {
    std::vector<Port> ports;
    std::vector<Net> nets;
    for (const auto& name : input_names) {
        ports.emplace_back(name, PortDirection::Input);
        nets.emplace_back(name, NetType::Input);
    }
    ports.emplace_back("y", PortDirection::Output);
    nets.emplace_back("y", NetType::Output);
    return Module("m", std::move(ports), std::move(nets),
                  {Instance("u0", std::move(type), std::move(connections))});
}

VerificationPortValues scalar_inputs(
    const std::vector<std::string>& names, unsigned vector) {
    VerificationPortValues result;
    for (size_t bit = 0; bit < names.size(); ++bit)
        result.emplace(names[bit], VerificationBits{
            static_cast<std::uint8_t>((vector >> bit) & 1U)});
    return result;
}

void test_binary_gate_truth_tables() {
    struct Gate {
        std::string type;
        bool (*reference)(bool, bool);
    };
    const std::vector<Gate> gates{
        {"AND2", [](bool a, bool b) { return a && b; }},
        {"OR2", [](bool a, bool b) { return a || b; }},
        {"XOR2", [](bool a, bool b) { return a != b; }},
    };
    for (const auto& gate : gates) {
        auto module = one_cell(gate.type, {"a", "b"},
            {{"A", "a"}, {"B", "b"}, {"Y", "y"}});
        for (unsigned vector = 0; vector < 4; ++vector) {
            const bool a = (vector & 1U) != 0;
            const bool b = (vector & 2U) != 0;
            const auto result = evaluate_combinational_netlist(
                module, scalar_inputs({"a", "b"}, vector));
            require(result.succeeded(), "binary gate evaluation failed");
            require(result.outputs.at("y").at(0) == gate.reference(a, b),
                    "binary gate differed from its independently authored truth table");
        }
    }
}

void test_unary_mux_and_constant_truth_tables() {
    for (const auto& [type, reference] : std::vector<std::pair<std::string,
            std::uint8_t (*)(std::uint8_t)>>{
            {"INV", [](std::uint8_t a) { return static_cast<std::uint8_t>(!a); }},
            {"BUF", [](std::uint8_t a) { return a; }}}) {
        auto module = one_cell(type, {"a"}, {{"A", "a"}, {"Y", "y"}});
        for (std::uint8_t a = 0; a < 2; ++a) {
            const auto result = evaluate_combinational_netlist(
                module, {{"a", {a}}});
            require(result.succeeded() && result.outputs.at("y").at(0) == reference(a),
                    "unary cell truth table mismatch");
        }
    }

    auto mux = one_cell("MUX2", {"a0", "a1", "s"},
        {{"A0", "a0"}, {"A1", "a1"}, {"S", "s"}, {"Y", "y"}});
    for (unsigned vector = 0; vector < 8; ++vector) {
        const auto input = scalar_inputs({"a0", "a1", "s"}, vector);
        const auto result = evaluate_combinational_netlist(mux, input);
        require(result.succeeded(), "MUX2 evaluation failed");
        const auto expected = input.at("s").at(0)
                                  ? input.at("a1").at(0)
                                  : input.at("a0").at(0);
        require(result.outputs.at("y").at(0) == expected,
                "MUX2 must select A1 when S is one and A0 when S is zero");
    }

    for (const auto& [type, expected] :
         std::vector<std::pair<std::string, std::uint8_t>>{{"TIEHI", 1},
                                                           {"TIELO", 0}}) {
        auto module = one_cell(type, {}, {{"Y", "y"}});
        const auto result = evaluate_combinational_netlist(module, {});
        require(result.succeeded() && result.outputs.at("y").at(0) == expected,
                "tie-cell truth table mismatch");
    }
}

void test_topological_fanout_and_declared_port_order() {
    Module chain("chain",
        {Port("a", PortDirection::Input), Port("b", PortDirection::Input),
         Port("y", PortDirection::Output), Port("z", PortDirection::Output)},
        {Net("a", NetType::Input), Net("b", NetType::Input),
         Net("n", NetType::Internal), Net("y", NetType::Output),
         Net("z", NetType::Output)},
        {Instance("late", "BUF", {{"A", "n"}, {"Y", "y"}}),
         Instance("gate", "AND2", {{"A", "a"}, {"B", "b"}, {"Y", "n"}}),
         Instance("fanout", "BUF", {{"A", "n"}, {"Y", "z"}})});
    const auto chain_result = evaluate_combinational_netlist(
        chain, {{"a", {1}}, {"b", {1}}});
    require(chain_result.succeeded(), "out-of-order cells or fanout did not evaluate");
    require(chain_result.outputs.at("y").at(0) == 1 &&
                chain_result.outputs.at("z").at(0) == 1,
            "fanout outputs do not share the driven value");

    Module ascending("ascending",
        {Port("a", PortDirection::Input, 3, 0, 2),
         Port("y", PortDirection::Output, 3, 0, 2)},
        {Net("a[0]", NetType::Input), Net("a[1]", NetType::Input),
         Net("a[2]", NetType::Input), Net("y[0]", NetType::Output),
         Net("y[1]", NetType::Output), Net("y[2]", NetType::Output)},
        {Instance("b0", "BUF", {{"A", "a[2]"}, {"Y", "y[2]"}}),
         Instance("b1", "BUF", {{"A", "a[1]"}, {"Y", "y[1]"}}),
         Instance("b2", "BUF", {{"A", "a[0]"}, {"Y", "y[0]"}})});
    require(module_port_bit_nets(ascending, ascending.get_ports().front()) ==
                std::vector<std::string>({"a[2]", "a[1]", "a[0]"}),
            "ascending port's LSB must be its declared right bound");
    const auto vector_result = evaluate_combinational_netlist(
        ascending, {{"a", {1, 0, 1}}});
    require(vector_result.succeeded() &&
                vector_result.outputs.at("y") == VerificationBits({1, 0, 1}),
            "packed port bits were not mapped in LSB-first declared-range order");

    Module descending("descending",
        {Port("a", PortDirection::Input, 3, 2, 0),
         Port("y", PortDirection::Output, 3, 2, 0)},
        {Net("a[0]", NetType::Input), Net("a[1]", NetType::Input),
         Net("a[2]", NetType::Input), Net("y[0]", NetType::Output),
         Net("y[1]", NetType::Output), Net("y[2]", NetType::Output)},
        {Instance("b0", "BUF", {{"A", "a[0]"}, {"Y", "y[0]"}}),
         Instance("b1", "BUF", {{"A", "a[1]"}, {"Y", "y[1]"}}),
         Instance("b2", "BUF", {{"A", "a[2]"}, {"Y", "y[2]"}})});
    require(module_port_bit_nets(descending, descending.get_ports().front()) ==
                std::vector<std::string>({"a[0]", "a[1]", "a[2]"}),
            "descending port's LSB must be its declared right bound");
    const auto descending_result = evaluate_combinational_netlist(
        descending, {{"a", {0, 1, 1}}});
    require(descending_result.succeeded() &&
                descending_result.outputs.at("y") == VerificationBits({0, 1, 1}),
            "descending packed range did not preserve LSB-first bit order");
}

void test_explicit_sequential_cutpoints() {
    Module module("state",
        {Port("clk", PortDirection::Input), Port("d", PortDirection::Input),
         Port("q", PortDirection::Output)},
        {Net("clk", NetType::Input), Net("d", NetType::Input),
         Net("state_q", NetType::Internal), Net("q", NetType::Output)},
        {Instance("ff", "DFF", {{"D", "d"}, {"CLK", "clk"},
                                  {"Y", "state_q"}}),
         Instance("out", "BUF", {{"A", "state_q"}, {"Y", "q"}})});
    const auto no_cutpoint = evaluate_combinational_netlist(
        module, {{"clk", {0}}, {"d", {1}}});
    require(!no_cutpoint.succeeded() &&
                no_cutpoint.status == NetlistVerificationStatus::UnsupportedCell,
            "sequential cells must be unsupported without explicit cutpoint mode");

    CombinationalEvaluationOptions options;
    options.allow_sequential_cutpoints = true;
    options.cutpoint_values.emplace("state_q", 1);
    const auto result = evaluate_combinational_netlist(
        module, {{"clk", {0}}, {"d", {0}}}, options);
    require(result.succeeded() && result.outputs.at("q").at(0) == 1,
            "explicit DFF cutpoint did not propagate through its combinational fanout");
    options.cutpoint_values["d"] = 1;
    const auto invalid_cutpoint = evaluate_combinational_netlist(
        module, {{"clk", {0}}, {"d", {0}}}, options);
    require(!invalid_cutpoint.succeeded(),
            "cutpoint values must identify actual sequential output nets");
}

void expect_status(const Module& module, const VerificationPortValues& inputs,
                   NetlistVerificationStatus expected,
                   std::string_view description,
                   const CombinationalEvaluationOptions& options = {}) {
    const auto result = evaluate_combinational_netlist(module, inputs, options);
    require(result.status == expected, description);
    require(!result.diagnostics.empty(), "failure result must include a diagnostic");
}

void test_malformed_netlists_and_limits() {
    const std::vector<Port> scalar_ports{
        Port("a", PortDirection::Input), Port("y", PortDirection::Output)};
    const std::vector<Net> scalar_nets{Net("a", NetType::Input),
                                       Net("y", NetType::Output)};
    expect_status(Module("missing", scalar_ports, scalar_nets,
                         {Instance("u", "BUF", {{"A", "a"}, {"Y", "y"}})}),
                  {}, NetlistVerificationStatus::InvalidStimulus,
                  "missing input stimulus must be rejected");
    expect_status(Module("four_state", scalar_ports, scalar_nets,
                         {Instance("u", "BUF", {{"A", "a"}, {"Y", "y"}})}),
                  {{"a", {2}}}, NetlistVerificationStatus::InvalidStimulus,
                  "non-two-state input must be rejected");
    expect_status(Module("bad_pins", scalar_ports, scalar_nets,
                         {Instance("u", "AND2", {{"A", "a"}, {"Y", "y"}})}),
                  {{"a", {1}}}, NetlistVerificationStatus::InvalidNetlist,
                  "missing generic cell pins must be diagnosed");
    expect_status(Module("parameters", scalar_ports, scalar_nets,
                         {Instance("u", "BUF", {{"A", "a"}, {"Y", "y"}},
                                   std::nullopt, {{"WIDTH", "1"}})}),
                  {{"a", {1}}}, NetlistVerificationStatus::InvalidNetlist,
                  "unexpected generic-cell parameters must be diagnosed");
    expect_status(Module("unsupported", scalar_ports, scalar_nets,
                         {Instance("u", "NAND2", {{"A", "a"}, {"B", "a"},
                                                    {"Y", "y"}})}),
                  {{"a", {1}}}, NetlistVerificationStatus::UnsupportedCell,
                  "unsupported generic cell must be diagnosed");
    expect_status(Module("undeclared", scalar_ports, scalar_nets,
                         {Instance("u", "BUF", {{"A", "missing"}, {"Y", "y"}})}),
                  {{"a", {1}}}, NetlistVerificationStatus::UndrivenNet,
                  "undeclared source net must be diagnosed");
    expect_status(Module("undriven", scalar_ports, scalar_nets, {}),
                  {{"a", {1}}}, NetlistVerificationStatus::UndrivenNet,
                  "undriven output must be diagnosed");

    const std::vector<Net> loop_nets{
        Net("a", NetType::Input), Net("loop", NetType::Internal),
        Net("y", NetType::Output)};
    expect_status(Module("cycle", scalar_ports, loop_nets,
                         {Instance("inv", "INV", {{"A", "loop"}, {"Y", "loop"}}),
                          Instance("out", "BUF", {{"A", "loop"}, {"Y", "y"}})}),
                  {{"a", {0}}}, NetlistVerificationStatus::CombinationalCycle,
                  "combinational cycle must be detected");

    const std::vector<Net> driven_nets{
        Net("a", NetType::Input), Net("y", NetType::Output)};
    expect_status(Module("drivers", scalar_ports, driven_nets,
                         {Instance("first", "BUF", {{"A", "a"}, {"Y", "y"}}),
                          Instance("second", "INV", {{"A", "a"}, {"Y", "y"}})}),
                  {{"a", {0}}}, NetlistVerificationStatus::MultipleDrivers,
                  "multiple drivers must be detected");

    expect_status(Module("duplicate_nets", scalar_ports,
                         {Net("a", NetType::Input), Net("a", NetType::Input),
                          Net("y", NetType::Output)},
                         {Instance("u", "BUF", {{"A", "a"}, {"Y", "y"}})}),
                  {{"a", {1}}}, NetlistVerificationStatus::InvalidNetlist,
                  "duplicate net declarations must be diagnosed");
    expect_status(Module("duplicate_instances", scalar_ports, scalar_nets,
                         {Instance("u", "BUF", {{"A", "a"}, {"Y", "y"}}),
                          Instance("u", "INV", {{"A", "a"}, {"Y", "y"}})}),
                  {{"a", {1}}}, NetlistVerificationStatus::InvalidNetlist,
                  "duplicate instance names must be diagnosed");

    auto limited = one_cell("BUF", {"a"}, {{"A", "a"}, {"Y", "y"}});
    CombinationalEvaluationOptions options;
    options.max_cell_evaluations = 0;
    expect_status(limited, {{"a", {1}}}, NetlistVerificationStatus::ResourceLimit,
                  "cell evaluation limit must be enforced", options);
}

} // namespace

int main() {
    try {
        test_binary_gate_truth_tables();
        test_unary_mux_and_constant_truth_tables();
        test_topological_fanout_and_declared_port_order();
        test_explicit_sequential_cutpoints();
        test_malformed_netlists_and_limits();
    } catch (const std::exception& error) {
        std::cerr << "EDA netlist evaluator test failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
