#ifndef LABNEURA_EDA_NETLIST_H
#define LABNEURA_EDA_NETLIST_H

#include "labneura/eda/cell_library.h"
#include "labneura/eda/expression.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace labneura::eda {

namespace detail {
struct ParsedVerilogContext;
struct NativeVerilogProjectContext;
class NativeVerilogProjectProjector;
}

enum class PortDirection {
    Input,
    Output,
    InOut
};

enum class NetType {
    Input,
    Output,
    InOut,
    Internal
};

class Port {
public:
    Port(std::string name, PortDirection direction, std::size_t width = 1,
         std::optional<std::int64_t> left_index = std::nullopt,
         std::optional<std::int64_t> right_index = std::nullopt,
         std::optional<std::int64_t> unpacked_left_index = std::nullopt,
         std::optional<std::int64_t> unpacked_right_index = std::nullopt,
         std::size_t packed_width = 0)
        : name_(std::move(name)),
          direction_(direction),
          width_(width),
          left_index_(left_index),
          right_index_(right_index),
          unpacked_left_index_(unpacked_left_index),
          unpacked_right_index_(unpacked_right_index),
          packed_width_(packed_width == 0 ? width : packed_width) {}

    const std::string& get_name() const noexcept { return name_; }
    PortDirection get_direction() const noexcept { return direction_; }
    std::size_t get_width() const noexcept { return width_; }
    const std::optional<std::int64_t>& get_left_index() const noexcept {
        return left_index_;
    }
    const std::optional<std::int64_t>& get_right_index() const noexcept {
        return right_index_;
    }
    const std::optional<std::int64_t>& get_unpacked_left_index() const noexcept {
        return unpacked_left_index_;
    }
    const std::optional<std::int64_t>& get_unpacked_right_index() const noexcept {
        return unpacked_right_index_;
    }
    std::size_t get_packed_width() const noexcept { return packed_width_; }

private:
    std::string name_;
    PortDirection direction_;
    std::size_t width_;
    std::optional<std::int64_t> left_index_;
    std::optional<std::int64_t> right_index_;
    std::optional<std::int64_t> unpacked_left_index_;
    std::optional<std::int64_t> unpacked_right_index_;
    std::size_t packed_width_;
};

class Net {
public:
    Net(std::string name, NetType kind, std::size_t width = 1,
        std::optional<std::int64_t> left_index = std::nullopt,
        std::optional<std::int64_t> right_index = std::nullopt,
        std::optional<std::int64_t> unpacked_left_index = std::nullopt,
        std::optional<std::int64_t> unpacked_right_index = std::nullopt,
        std::size_t packed_width = 0)
        : name_(std::move(name)),
          kind_(kind),
          width_(width),
          left_index_(left_index),
          right_index_(right_index),
          unpacked_left_index_(unpacked_left_index),
          unpacked_right_index_(unpacked_right_index),
          packed_width_(packed_width == 0 ? width : packed_width) {}

    const std::string& get_name() const noexcept { return name_; }
    NetType get_kind() const noexcept { return kind_; }
    std::size_t get_width() const noexcept { return width_; }
    const std::optional<std::int64_t>& get_left_index() const noexcept {
        return left_index_;
    }
    const std::optional<std::int64_t>& get_right_index() const noexcept {
        return right_index_;
    }
    const std::optional<std::int64_t>& get_unpacked_left_index() const noexcept {
        return unpacked_left_index_;
    }
    const std::optional<std::int64_t>& get_unpacked_right_index() const noexcept {
        return unpacked_right_index_;
    }
    std::size_t get_packed_width() const noexcept { return packed_width_; }

private:
    std::string name_;
    NetType kind_;
    std::size_t width_;
    std::optional<std::int64_t> left_index_;
    std::optional<std::int64_t> right_index_;
    std::optional<std::int64_t> unpacked_left_index_;
    std::optional<std::int64_t> unpacked_right_index_;
    std::size_t packed_width_;
};

class PinConnection {
public:
    PinConnection(std::string pin, std::string net)
        : pin_(std::move(pin)), net_(std::move(net)) {}

    const std::string& get_pin() const noexcept { return pin_; }
    const std::string& get_net() const noexcept { return net_; }

private:
    std::string pin_;
    std::string net_;
};

struct UserDefinedPrimitiveRow {
    std::string inputs;
    char output;
};

class Instance {
public:
    Instance(std::string name, std::string cell_type,
             std::vector<PinConnection> connections,
             std::optional<StandardCell> cell = std::nullopt,
             std::map<std::string, std::string> parameters = {},
             bool user_defined_primitive = false,
             std::vector<UserDefinedPrimitiveRow> primitive_table = {},
             bool sequential_primitive = false)
        : name_(std::move(name)),
          cell_type_(std::move(cell_type)),
          connections_(std::move(connections)),
          cell_(std::move(cell)),
          parameters_(std::move(parameters)),
          user_defined_primitive_(user_defined_primitive),
          primitive_table_(std::move(primitive_table)),
          sequential_primitive_(sequential_primitive) {}

    const std::string& get_name() const noexcept { return name_; }
    const std::string& get_cell_type() const noexcept { return cell_type_; }
    const std::vector<PinConnection>& get_connections() const noexcept {
        return connections_;
    }
    const std::optional<StandardCell>& get_cell() const noexcept { return cell_; }
    // Cell parameters, e.g. the truth table "INIT" of an FPGA LUT.
    const std::map<std::string, std::string>& get_parameters() const noexcept {
        return parameters_;
    }
    bool is_user_defined_primitive() const noexcept {
        return user_defined_primitive_;
    }
    const std::vector<UserDefinedPrimitiveRow>& get_primitive_table() const
        noexcept {
        return primitive_table_;
    }
    bool is_sequential_primitive() const noexcept {
        return sequential_primitive_;
    }

private:
    std::string name_;
    std::string cell_type_;
    std::vector<PinConnection> connections_;
    std::optional<StandardCell> cell_;
    std::map<std::string, std::string> parameters_;
    bool user_defined_primitive_ = false;
    std::vector<UserDefinedPrimitiveRow> primitive_table_;
    bool sequential_primitive_ = false;
};

class ContinuousAssignment {
public:
    ContinuousAssignment(Expression lhs, Expression rhs)
        : lhs_(std::move(lhs)), rhs_(std::move(rhs)) {}

    const std::string& get_lhs() const noexcept { return lhs_.get_source(); }
    const std::string& get_rhs() const noexcept { return rhs_.get_source(); }
    const Expression& get_lhs_expression() const noexcept { return lhs_; }
    const Expression& get_rhs_expression() const noexcept { return rhs_; }

private:
    Expression lhs_;
    Expression rhs_;
};

enum class MemoryReadDuringWrite {
    ReadBeforeWrite
};

// Technology-independent synchronous single-port memory. This is deliberately
// retained above the bit-cell layer so FPGA BRAM and ASIC SRAM mapping can
// consume it without first expanding storage into flip-flops and muxes.
class Memory {
public:
    Memory(std::string name, std::size_t word_width, std::size_t depth,
           std::size_t address_width, std::string clock,
           std::string write_enable, std::vector<std::string> write_address,
           std::vector<std::string> write_data,
           std::vector<std::string> read_address,
           std::vector<std::string> read_data,
           MemoryReadDuringWrite read_during_write =
               MemoryReadDuringWrite::ReadBeforeWrite)
        : name_(std::move(name)), word_width_(word_width), depth_(depth),
          address_width_(address_width), clock_(std::move(clock)),
          write_enable_(std::move(write_enable)),
          write_address_(std::move(write_address)), write_data_(std::move(write_data)),
          read_address_(std::move(read_address)), read_data_(std::move(read_data)),
          read_during_write_(read_during_write) {}

    const std::string& get_name() const noexcept { return name_; }
    std::size_t get_word_width() const noexcept { return word_width_; }
    std::size_t get_depth() const noexcept { return depth_; }
    std::size_t get_address_width() const noexcept { return address_width_; }
    const std::string& get_clock() const noexcept { return clock_; }
    const std::string& get_write_enable() const noexcept { return write_enable_; }
    const std::vector<std::string>& get_write_address() const noexcept { return write_address_; }
    const std::vector<std::string>& get_write_data() const noexcept { return write_data_; }
    const std::vector<std::string>& get_read_address() const noexcept { return read_address_; }
    const std::vector<std::string>& get_read_data() const noexcept { return read_data_; }
    MemoryReadDuringWrite get_read_during_write() const noexcept {
        return read_during_write_;
    }

private:
    std::string name_;
    std::size_t word_width_ = 0;
    std::size_t depth_ = 0;
    std::size_t address_width_ = 0;
    std::string clock_;
    std::string write_enable_;
    std::vector<std::string> write_address_;
    std::vector<std::string> write_data_;
    std::vector<std::string> read_address_;
    std::vector<std::string> read_data_;
    MemoryReadDuringWrite read_during_write_ = MemoryReadDuringWrite::ReadBeforeWrite;
};

class Project;

class Module {
public:
    Module() = default;
    Module(std::string name, std::vector<Port> ports, std::vector<Net> nets,
           std::vector<Instance> instances,
           std::vector<ContinuousAssignment> assignments = {},
           bool behavioral = false, std::string hierarchy_instance_name = {},
           std::vector<Module> child_modules = {}, std::vector<Memory> memories = {})
        : name_(std::move(name)),
          ports_(std::move(ports)),
          nets_(std::move(nets)),
          instances_(std::move(instances)),
          assignments_(std::move(assignments)),
          behavioral_(behavioral),
          hierarchy_instance_name_(std::move(hierarchy_instance_name)),
          child_modules_(std::move(child_modules)), memories_(std::move(memories)) {}

    const std::string& get_name() const noexcept { return name_; }
    // Empty for a standalone module definition; otherwise the instance name
    // that connects this hierarchy node to its parent.
    const std::string& get_hierarchy_instance_name() const noexcept {
        return hierarchy_instance_name_;
    }
    const std::vector<Port>& get_ports() const noexcept { return ports_; }
    const std::vector<Net>& get_nets() const noexcept { return nets_; }
    const std::vector<Instance>& get_instances() const noexcept {
        return instances_;
    }
    // Elaborated child module instances, in the same scope as get_instances().
    const std::vector<Module>& get_child_modules() const noexcept {
        return child_modules_;
    }
    const std::vector<ContinuousAssignment>& get_assignments() const noexcept {
        return assignments_;
    }
    const std::vector<Memory>& get_memories() const noexcept { return memories_; }
    std::size_t get_port_count() const noexcept { return ports_.size(); }
    std::size_t get_wire_count() const noexcept {
        std::size_t count = 0;
        for (const auto& net : nets_) {
            if (net.get_kind() == NetType::Internal) ++count;
        }
        return count;
    }
    std::size_t get_instance_count() const noexcept { return instances_.size(); }
    // True when the RTL contains always blocks that generic_map must lower.
    bool is_behavioral() const noexcept { return behavioral_; }
private:
    std::string name_;
    std::vector<Port> ports_;
    std::vector<Net> nets_;
    std::vector<Instance> instances_;
    std::vector<ContinuousAssignment> assignments_;
    bool behavioral_ = false;
    std::string hierarchy_instance_name_;
    std::vector<Module> child_modules_;
    std::vector<Memory> memories_;
    std::shared_ptr<detail::ParsedVerilogContext> parsed_context_;
    std::shared_ptr<const detail::NativeVerilogProjectContext> native_context_;

    friend class detail::NativeVerilogProjectProjector;
    friend Module parse_verilog(const std::string& path);
    friend Module read_verilog(const std::string& path);
    friend Project parse_verilog_directory(const std::string& directory);
    friend Module generic_map(const Module& rtl_module);
    friend Module generic_map(
        const Project& project, const std::string& top,
        const std::map<std::string, std::string>& parameters);
};

class Project {
public:
    Project() = default;

    const std::vector<Module>& get_modules() const noexcept { return modules_; }
    std::size_t get_module_count() const noexcept { return modules_.size(); }
    std::size_t get_total_port_count() const noexcept {
        std::size_t count = 0;
        for (const auto& module : modules_) count += module.get_port_count();
        return count;
    }
    std::size_t get_total_wire_count() const noexcept {
        std::size_t count = 0;
        for (const auto& module : modules_) count += module.get_wire_count();
        return count;
    }
    std::size_t get_total_instance_count() const noexcept {
        std::size_t count = 0;
        for (const auto& module : modules_) count += module.get_instance_count();
        return count;
    }
    const std::string& get_ast_json() const noexcept { return ast_json_; }
    const std::vector<std::string>& get_source_files() const noexcept {
        return source_files_;
    }

private:
    std::vector<Module> modules_;
    std::string ast_json_;
    std::vector<std::string> source_files_;
    std::shared_ptr<detail::ParsedVerilogContext> parsed_context_;
    std::shared_ptr<const detail::NativeVerilogProjectContext> native_context_;

    friend class detail::NativeVerilogProjectProjector;
    friend Project parse_verilog_directory(const std::string& directory);
    friend Module generic_map(
        const Project& project, const std::string& top,
        const std::map<std::string, std::string>& parameters);
    friend Module generic_map(const Module& rtl_module);
};

Module parse_verilog(const std::string& path);
Project parse_verilog_directory(const std::string& directory);
// Reads a structural Verilog netlist into the same parsed Module model.
Module read_verilog(const std::string& path);
// Lowers a parsed module, or the hierarchy below `top` in a parsed project,
// into technology-independent generic cells. `parameters` overrides top-level
// module parameters (for example {"ENABLE_MUL": "1"}).
Module generic_map(const Module& rtl_module);
Module generic_map(const Project& project, const std::string& top = "",
                   const std::map<std::string, std::string>& parameters = {});
Module tech_map(const Module& generic_module,
                const StandardCellLibrary& cell_library);
Module tech_map(const Project& project, const StandardCellLibrary& cell_library,
                const std::string& top = "",
                const std::map<std::string, std::string>& parameters = {});
// Maps a generic netlist onto `lut_size`-input LUTs (2..6) with a depth-optimal,
// area-recovering cut-based mapper. Flops, latches and tristates pass through.
// LUT cells are named LUT<n>, use pins I0..I<n-1> and O, and carry the truth
// table as hex parameter "INIT" (bit i = output for input pattern i, I0 = LSB).
// Every mapped instance carries its `cell` (area, pins, timing arcs) taken from
// `library`, which defaults to the built-in generic FPGA architecture model.
const StandardCellLibrary& get_default_fpga_library();
Module fpga_map(const Module& generic_module, std::size_t lut_size = 4,
                const StandardCellLibrary& library = get_default_fpga_library());
Module fpga_map(const Project& project, const std::string& top = "",
                const std::map<std::string, std::string>& parameters = {},
                std::size_t lut_size = 4,
                const StandardCellLibrary& library = get_default_fpga_library());
// Writes a LUT netlist as BLIF for tools such as Yosys, ABC or VPR.
std::string to_blif(const Module& module);
void write_blif(const Module& module, const std::string& path);
std::string to_json(const Module& module);
void write_json(const Module& module, const std::string& path);
Module read_json(const std::string& path);
std::string to_verilog(const Module& module);
void write_verilog(const Module& module, const std::string& path);
// Graphviz DOT view: cells and ports are nodes, nets are driver-to-sink edges.
std::string to_dot(const Module& module);
void write_dot(const Module& module, const std::string& path);

}  // namespace labneura::eda

#endif
