#ifndef LABNEURA_EDA_CELL_LIBRARY_H
#define LABNEURA_EDA_CELL_LIBRARY_H

#include "labneura/tensor.h"

#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace labneura::eda {

class LookupTable {
public:
    LookupTable(std::string template_name, std::vector<double> index_1,
                std::vector<double> index_2,
                std::vector<std::vector<double>> values)
        : template_name_(std::move(template_name)),
          index_1_(std::move(index_1)),
          index_2_(std::move(index_2)),
          values_(std::move(values)) {}

    const std::string& get_template_name() const noexcept {
        return template_name_;
    }
    const std::vector<double>& get_index_1() const noexcept { return index_1_; }
    const std::vector<double>& get_index_2() const noexcept { return index_2_; }
    const std::vector<std::vector<double>>& get_values() const noexcept {
        return values_;
    }

    // 2-D FP32 tensor (rows x columns) of the table values.
    Tensor to_tensor() const {
        std::vector<std::vector<float>> rows;
        rows.reserve(values_.size());
        for (const auto& row : values_) {
            rows.emplace_back(row.begin(), row.end());
        }
        return Tensor(rows);
    }

private:
    std::string template_name_;
    std::vector<double> index_1_;
    std::vector<double> index_2_;
    std::vector<std::vector<double>> values_;
};

class TimingArc {
public:
    TimingArc(std::string related_pin, std::string timing_type,
              std::string timing_sense,
              std::optional<LookupTable> cell_rise,
              std::optional<LookupTable> cell_fall,
              std::optional<LookupTable> rise_transition,
              std::optional<LookupTable> fall_transition,
              std::optional<LookupTable> rise_constraint,
              std::optional<LookupTable> fall_constraint)
        : related_pin_(std::move(related_pin)),
          timing_type_(std::move(timing_type)),
          timing_sense_(std::move(timing_sense)),
          cell_rise_(std::move(cell_rise)),
          cell_fall_(std::move(cell_fall)),
          rise_transition_(std::move(rise_transition)),
          fall_transition_(std::move(fall_transition)),
          rise_constraint_(std::move(rise_constraint)),
          fall_constraint_(std::move(fall_constraint)) {}

    const std::string& get_related_pin() const noexcept { return related_pin_; }
    const std::string& get_timing_type() const noexcept { return timing_type_; }
    const std::string& get_timing_sense() const noexcept { return timing_sense_; }
    const std::optional<LookupTable>& get_cell_rise() const noexcept {
        return cell_rise_;
    }
    const std::optional<LookupTable>& get_cell_fall() const noexcept {
        return cell_fall_;
    }
    const std::optional<LookupTable>& get_rise_transition() const noexcept {
        return rise_transition_;
    }
    const std::optional<LookupTable>& get_fall_transition() const noexcept {
        return fall_transition_;
    }
    const std::optional<LookupTable>& get_rise_constraint() const noexcept {
        return rise_constraint_;
    }
    const std::optional<LookupTable>& get_fall_constraint() const noexcept {
        return fall_constraint_;
    }

private:
    std::string related_pin_;
    std::string timing_type_;
    std::string timing_sense_;
    std::optional<LookupTable> cell_rise_;
    std::optional<LookupTable> cell_fall_;
    std::optional<LookupTable> rise_transition_;
    std::optional<LookupTable> fall_transition_;
    std::optional<LookupTable> rise_constraint_;
    std::optional<LookupTable> fall_constraint_;
};

class CellPin {
public:
    CellPin(std::string name, std::string direction, std::string function,
            std::optional<double> capacitance, std::vector<TimingArc> timing_arcs)
        : name_(std::move(name)),
          direction_(std::move(direction)),
          function_(std::move(function)),
          capacitance_(capacitance),
          timing_arcs_(std::move(timing_arcs)) {}

    const std::string& get_name() const noexcept { return name_; }
    const std::string& get_direction() const noexcept { return direction_; }
    const std::string& get_function() const noexcept { return function_; }
    const std::optional<double>& get_capacitance() const noexcept {
        return capacitance_;
    }
    const std::vector<TimingArc>& get_timing_arcs() const noexcept {
        return timing_arcs_;
    }

private:
    std::string name_;
    std::string direction_;
    std::string function_;
    std::optional<double> capacitance_;
    std::vector<TimingArc> timing_arcs_;
};

class StandardCell {
public:
    StandardCell(std::string name, std::optional<double> area,
                 std::vector<CellPin> pins)
        : name_(std::move(name)), area_(area), pins_(std::move(pins)) {}

    const std::string& get_name() const noexcept { return name_; }
    const std::optional<double>& get_area() const noexcept { return area_; }
    const std::vector<CellPin>& get_pins() const noexcept { return pins_; }
    const CellPin* find_pin(const std::string& name) const noexcept {
        for (const auto& pin : pins_) {
            if (pin.get_name() == name) return &pin;
        }
        return nullptr;
    }

private:
    std::string name_;
    std::optional<double> area_;
    std::vector<CellPin> pins_;
};

class StandardCellLibrary {
public:
    StandardCellLibrary(std::string name, std::string time_unit,
                        std::string capacitance_unit,
                        std::vector<StandardCell> cells)
        : name_(std::move(name)),
          time_unit_(std::move(time_unit)),
          capacitance_unit_(std::move(capacitance_unit)),
          cells_(std::move(cells)) {}

    const std::string& get_name() const noexcept { return name_; }
    const std::string& get_time_unit() const noexcept { return time_unit_; }
    const std::string& get_capacitance_unit() const noexcept {
        return capacitance_unit_;
    }
    const std::vector<StandardCell>& get_cells() const noexcept { return cells_; }
    std::size_t get_cell_count() const noexcept { return cells_.size(); }
    const StandardCell* find_cell(const std::string& name) const noexcept {
        for (const auto& cell : cells_) {
            if (cell.get_name() == name) return &cell;
        }
        return nullptr;
    }

private:
    std::string name_;
    std::string time_unit_;
    std::string capacitance_unit_;
    std::vector<StandardCell> cells_;
};

StandardCellLibrary parse_liberty(const std::string& path);
StandardCellLibrary parse_liberty_source(const std::string& source);
const StandardCellLibrary& get_default_cell_library();

}  // namespace labneura::eda

#endif
