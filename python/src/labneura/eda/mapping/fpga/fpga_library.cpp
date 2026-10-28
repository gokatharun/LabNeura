#include "labneura/eda/netlist.h"

#include <string>
#include <utility>
#include <vector>

namespace labneura::eda {
namespace {

// Illustrative architecture model, not a vendor characterisation: area is in
// configuration-bit equivalents, delays in ns, capacitances in pF, and routing
// delay is not modelled. Load a Liberty file to substitute real numbers.
constexpr double kPinCapacitance = 0.002;
constexpr double kClockToQ = 0.35;
constexpr double kSetup = 0.10;

LookupTable scalar(double value) {
    return LookupTable("scalar", {0.0}, {0.0}, {{value}});
}

TimingArc combinational_arc(const std::string& from, double delay) {
    return TimingArc(from, "combinational", "non_unate", scalar(delay), scalar(delay),
                     std::nullopt, std::nullopt, std::nullopt, std::nullopt);
}

TimingArc clock_to_q_arc(const std::string& clock) {
    return TimingArc(clock, "rising_edge", "non_unate", scalar(kClockToQ),
                     scalar(kClockToQ), std::nullopt, std::nullopt, std::nullopt,
                     std::nullopt);
}

TimingArc setup_arc(const std::string& clock) {
    return TimingArc(clock, "setup_rising", "non_unate", std::nullopt, std::nullopt,
                     std::nullopt, std::nullopt, scalar(kSetup), scalar(kSetup));
}

CellPin input_pin(const std::string& name, std::vector<TimingArc> arcs = {}) {
    return CellPin(name, "input", "", kPinCapacitance, std::move(arcs));
}

CellPin output_pin(const std::string& name, std::vector<TimingArc> arcs) {
    return CellPin(name, "output", "", std::nullopt, std::move(arcs));
}

StandardCell lut_cell(std::size_t inputs) {
    const double delay = 0.12 + 0.06 * static_cast<double>(inputs > 2 ? inputs - 2 : 0);
    std::vector<CellPin> pins;
    std::vector<TimingArc> arcs;
    for (std::size_t i = 0; i < inputs; ++i) {
        pins.push_back(input_pin("I" + std::to_string(i)));
        arcs.push_back(combinational_arc("I" + std::to_string(i), delay));
    }
    pins.push_back(output_pin("O", std::move(arcs)));
    return StandardCell("LUT" + std::to_string(inputs),
                        static_cast<double>(1U << inputs), std::move(pins));
}

StandardCell flop_cell(const std::string& name, double area,
                       const std::vector<std::string>& controls) {
    std::vector<CellPin> pins = {input_pin("D", {setup_arc("CLK")}),
                                 input_pin("CLK")};
    for (const auto& control : controls) pins.push_back(input_pin(control));
    pins.push_back(output_pin("Y", {clock_to_q_arc("CLK")}));
    return StandardCell(name, area, std::move(pins));
}

}  // namespace

const StandardCellLibrary& get_default_fpga_library() {
    static const StandardCellLibrary library = [] {
        std::vector<StandardCell> cells;
        for (std::size_t inputs = 1; inputs <= 6; ++inputs) cells.push_back(lut_cell(inputs));
        cells.push_back(flop_cell("DFF", 16.0, {}));
        cells.push_back(flop_cell("DFFR", 18.0, {"RESET_B"}));
        cells.push_back(flop_cell("DFFS", 18.0, {"SET_B"}));
        cells.push_back(flop_cell("DFFSR", 20.0, {"RESET_B", "SET_B"}));
        cells.push_back(StandardCell(
            "LATCH", 12.0,
            {input_pin("D", {setup_arc("GATE")}), input_pin("GATE"),
             output_pin("Y", {combinational_arc("D", 0.2)})}));
        cells.push_back(StandardCell(
            "TRIBUF", 4.0,
            {input_pin("A"), input_pin("TE_B"),
             output_pin("Y", {combinational_arc("A", 0.15)})}));
        cells.push_back(StandardCell("TIEHI", 0.0, {output_pin("Y", {})}));
        cells.push_back(StandardCell("TIELO", 0.0, {output_pin("Y", {})}));
        return StandardCellLibrary("labneura_generic_fpga", "1ns", "1pf", std::move(cells));
    }();
    return library;
}

}  // namespace labneura::eda
