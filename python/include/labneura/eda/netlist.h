#ifndef LABNEURA_EDA_NETLIST_H
#define LABNEURA_EDA_NETLIST_H

#include "labneura/eda/cell_library.h"
#include "labneura/eda/expression.h"

#include <cstddef>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace labneura::eda {

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
    Port(std::string name, PortDirection direction)
        : name_(std::move(name)), direction_(direction) {}

    const std::string& get_name() const noexcept { return name_; }
    PortDirection get_direction() const noexcept { return direction_; }

private:
    std::string name_;
    PortDirection direction_;
};

class Net {
public:
    Net(std::string name, NetType kind)
        : name_(std::move(name)), kind_(kind) {}

    const std::string& get_name() const noexcept { return name_; }
    NetType get_kind() const noexcept { return kind_; }

private:
    std::string name_;
    NetType kind_;
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

class Instance {
public:
    Instance(std::string name, std::string cell_type,
             std::vector<PinConnection> connections,
             std::optional<StandardCell> cell = std::nullopt)
        : name_(std::move(name)),
          cell_type_(std::move(cell_type)),
          connections_(std::move(connections)),
          cell_(std::move(cell)) {}

    const std::string& get_name() const noexcept { return name_; }
    const std::string& get_cell_type() const noexcept { return cell_type_; }
    const std::vector<PinConnection>& get_connections() const noexcept {
        return connections_;
    }
    const std::optional<StandardCell>& get_cell() const noexcept { return cell_; }

private:
    std::string name_;
    std::string cell_type_;
    std::vector<PinConnection> connections_;
    std::optional<StandardCell> cell_;
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

class Module {
public:
    Module() = default;
    Module(std::string name, std::vector<Port> ports, std::vector<Net> nets,
           std::vector<Instance> instances,
           std::vector<ContinuousAssignment> assignments = {},
           bool behavioral = false)
        : name_(std::move(name)),
          ports_(std::move(ports)),
          nets_(std::move(nets)),
          instances_(std::move(instances)),
          assignments_(std::move(assignments)),
          behavioral_(behavioral) {}

    const std::string& get_name() const noexcept { return name_; }
    const std::vector<Port>& get_ports() const noexcept { return ports_; }
    const std::vector<Net>& get_nets() const noexcept { return nets_; }
    const std::vector<Instance>& get_instances() const noexcept {
        return instances_;
    }
    const std::vector<ContinuousAssignment>& get_assignments() const noexcept {
        return assignments_;
    }
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
    // Source files a behavioral module was parsed from (set by parse_verilog).
    const std::vector<std::string>& get_source_files() const noexcept {
        return source_files_;
    }

private:
    std::string name_;
    std::vector<Port> ports_;
    std::vector<Net> nets_;
    std::vector<Instance> instances_;
    std::vector<ContinuousAssignment> assignments_;
    bool behavioral_ = false;
    std::vector<std::string> source_files_;

    friend Module parse_verilog_source(const std::string& source);
    friend Module parse_verilog(const std::string& path);
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

    friend Project parse_verilog_project(const std::string& directory);
};

Module parse_verilog_source(const std::string& source);
Module parse_verilog(const std::string& path);
Module parse_verilog_file(const std::string& path);
Project parse_verilog_project(const std::string& directory);
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
std::string to_json(const Module& module);
void write_json(const Module& module, const std::string& path);
// Graphviz DOT view: cells and ports are nodes, nets are driver-to-sink edges.
std::string to_dot(const Module& module);
void write_dot(const Module& module, const std::string& path);

}  // namespace labneura::eda

#endif
