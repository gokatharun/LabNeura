#include "eda_bindings.h"

#include "labneura/eda/cell_library.h"
#include "labneura/eda/netlist.h"

#include <pybind11/stl.h>

void bind_eda(pybind11::module_& labneura_module) {
    namespace py = pybind11;
    namespace eda = labneura::eda;

    labneura_module.attr("__path__") = py::list();
    py::module_ module = labneura_module.def_submodule(
        "eda", "Structural Verilog parsing and netlist interchange.");

    py::enum_<eda::PortDirection>(module, "PortDirection")
        .value("INPUT", eda::PortDirection::Input)
        .value("OUTPUT", eda::PortDirection::Output)
        .value("INOUT", eda::PortDirection::InOut);

    py::enum_<eda::NetType>(module, "NetType")
        .value("INPUT", eda::NetType::Input)
        .value("OUTPUT", eda::NetType::Output)
        .value("INOUT", eda::NetType::InOut)
        .value("INTERNAL", eda::NetType::Internal);

    py::enum_<eda::ExpressionKind>(module, "ExpressionKind")
        .value("IDENTIFIER", eda::ExpressionKind::Identifier)
        .value("LITERAL", eda::ExpressionKind::Literal)
        .value("UNARY", eda::ExpressionKind::Unary)
        .value("BINARY", eda::ExpressionKind::Binary)
        .value("CONDITIONAL", eda::ExpressionKind::Conditional)
        .value("BIT_SELECT", eda::ExpressionKind::BitSelect)
        .value("RANGE_SELECT", eda::ExpressionKind::RangeSelect)
        .value("CONCATENATION", eda::ExpressionKind::Concatenation)
        .value("OPAQUE", eda::ExpressionKind::Opaque);

    py::class_<eda::Expression>(module, "Expression",
                                "A node in a Verilog expression tree.")
        .def_property_readonly("kind", &eda::Expression::get_kind)
        .def_property_readonly("value", &eda::Expression::get_value)
        .def_property_readonly("source", &eda::Expression::get_source)
        .def_property_readonly("operands", &eda::Expression::get_operands)
        .def_property_readonly("width", &eda::Expression::get_width)
        .def("get_kind", &eda::Expression::get_kind)
        .def("get_value", &eda::Expression::get_value)
        .def("get_source", &eda::Expression::get_source)
        .def("get_operands", &eda::Expression::get_operands)
        .def("get_width", &eda::Expression::get_width);

    py::class_<eda::Port>(module, "Port", "A module input or output port.")
        .def_property_readonly("name", &eda::Port::get_name, "Port identifier.")
        .def_property_readonly("direction", &eda::Port::get_direction,
                               "Port direction.")
        .def("get_name", &eda::Port::get_name)
        .def("get_direction", &eda::Port::get_direction);

    py::class_<eda::Net>(module, "Net", "A signal net declared in a module.")
        .def_property_readonly("name", &eda::Net::get_name, "Net identifier.")
        .def_property_readonly("kind", &eda::Net::get_kind, "Net type.")
        .def("get_name", &eda::Net::get_name)
        .def("get_kind", &eda::Net::get_kind);

    py::class_<eda::PinConnection>(
        module, "PinConnection", "A mapping from an instance pin to a net.")
        .def_property_readonly("pin", &eda::PinConnection::get_pin,
                               "Instance pin name.")
        .def_property_readonly("net", &eda::PinConnection::get_net,
                               "Connected net name.")
        .def("get_pin", &eda::PinConnection::get_pin)
        .def("get_net", &eda::PinConnection::get_net);

    py::class_<eda::Instance>(module, "Instance", "A gate instance in a module.")
        .def_property_readonly("name", &eda::Instance::get_name,
                               "Instance identifier.")
        .def_property_readonly("cell_type", &eda::Instance::get_cell_type,
                               "Gate type.")
        .def_property_readonly("cell", &eda::Instance::get_cell,
                               "Resolved standard-cell definition.")
        .def_property_readonly("connections", &eda::Instance::get_connections,
                               "Instance pin-to-net connections.")
        .def_property_readonly("parameters", &eda::Instance::get_parameters,
                               "Cell parameters such as a LUT's truth table INIT.")
        .def("get_name", &eda::Instance::get_name)
        .def("get_cell_type", &eda::Instance::get_cell_type)
        .def("get_cell", &eda::Instance::get_cell)
        .def("get_connections", &eda::Instance::get_connections);

    py::class_<eda::ContinuousAssignment>(
        module, "ContinuousAssignment",
        "A continuous assignment's left- and right-hand expressions.")
        .def_property_readonly("lhs", &eda::ContinuousAssignment::get_lhs,
                               "Left-hand expression.")
        .def_property_readonly("rhs", &eda::ContinuousAssignment::get_rhs,
                               "Right-hand expression.")
        .def_property_readonly("lhs_expression",
                               &eda::ContinuousAssignment::get_lhs_expression,
                               "Structured left-hand expression.")
        .def_property_readonly("rhs_expression",
                               &eda::ContinuousAssignment::get_rhs_expression,
                               "Structured right-hand expression.")
        .def("get_lhs", &eda::ContinuousAssignment::get_lhs)
        .def("get_rhs", &eda::ContinuousAssignment::get_rhs)
        .def("get_lhs_expression",
             &eda::ContinuousAssignment::get_lhs_expression)
        .def("get_rhs_expression",
             &eda::ContinuousAssignment::get_rhs_expression);

    py::class_<eda::Module>(module, "Module", "A parsed Verilog module.")
        .def_property_readonly("name", &eda::Module::get_name,
                               "Module identifier.")
        .def_property_readonly("ports", &eda::Module::get_ports,
                               "Module input and output ports.")
        .def_property_readonly("nets", &eda::Module::get_nets, "Module nets.")
        .def_property_readonly("is_behavioral", &eda::Module::is_behavioral,
                               "True when the module has always blocks that generic_map lowers.")
        .def_property_readonly("instances", &eda::Module::get_instances,
                               "Gate instances in the module.")
        .def_property_readonly("assignments", &eda::Module::get_assignments,
                               "Continuous assignments in the module.")
        .def("get_name", &eda::Module::get_name)
        .def("get_ports", &eda::Module::get_ports)
        .def("get_nets", &eda::Module::get_nets)
        .def("get_instances", &eda::Module::get_instances)
        .def("get_assignments", &eda::Module::get_assignments)
        .def("dump_graph",
             [](const eda::Module& self, const std::string& path) {
                 if (path.empty()) return eda::to_dot(self);
                 eda::write_dot(self, path);
                 return std::string();
             },
             py::arg("path") = "",
             "Return the netlist as Graphviz DOT, or write it to `path`.")
        .def_property_readonly("port_count", &eda::Module::get_port_count)
        .def_property_readonly("wire_count", &eda::Module::get_wire_count)
        .def_property_readonly("instance_count", &eda::Module::get_instance_count)
        .def("get_port_count", &eda::Module::get_port_count)
        .def("get_wire_count", &eda::Module::get_wire_count)
        .def("get_instance_count", &eda::Module::get_instance_count);

    py::class_<eda::Project>(module, "Project", "A parsed Verilog project.")
        .def_property_readonly("modules", &eda::Project::get_modules,
                               "Modules discovered in the project.")
        .def_property_readonly("module_count", &eda::Project::get_module_count)
        .def_property_readonly("total_port_count",
                               &eda::Project::get_total_port_count)
        .def_property_readonly("total_wire_count",
                               &eda::Project::get_total_wire_count)
        .def_property_readonly("total_instance_count",
                               &eda::Project::get_total_instance_count)
        .def_property_readonly("ast_json", &eda::Project::get_ast_json,
                               "Elaborated slang AST as JSON.")
        .def_property_readonly("source_files", &eda::Project::get_source_files,
                               "Verilog source files parsed for this project.")
        .def("get_modules", &eda::Project::get_modules)
        .def("get_module_count", &eda::Project::get_module_count)
        .def("get_total_port_count", &eda::Project::get_total_port_count)
        .def("get_total_wire_count", &eda::Project::get_total_wire_count)
        .def("get_total_instance_count", &eda::Project::get_total_instance_count)
        .def("get_ast_json", &eda::Project::get_ast_json)
        .def("get_source_files", &eda::Project::get_source_files);

    py::class_<eda::LookupTable>(module, "LookupTable",
                                 "A Liberty timing lookup table.")
        .def_property_readonly("template_name",
                               &eda::LookupTable::get_template_name)
        .def_property_readonly("index_1", &eda::LookupTable::get_index_1)
        .def_property_readonly("index_2", &eda::LookupTable::get_index_2)
        .def_property_readonly("values", &eda::LookupTable::get_values)
        .def("get_template_name", &eda::LookupTable::get_template_name)
        .def("get_index_1", &eda::LookupTable::get_index_1)
        .def("get_index_2", &eda::LookupTable::get_index_2)
        .def("get_values", &eda::LookupTable::get_values)
        .def("to_tensor", &eda::LookupTable::to_tensor,
             "Values as a flat row-major FP32 Tensor");

    py::class_<eda::TimingArc>(module, "TimingArc", "A Liberty timing arc.")
        .def_property_readonly("related_pin", &eda::TimingArc::get_related_pin)
        .def_property_readonly("timing_type", &eda::TimingArc::get_timing_type)
        .def_property_readonly("timing_sense", &eda::TimingArc::get_timing_sense)
        .def_property_readonly("cell_rise", &eda::TimingArc::get_cell_rise)
        .def_property_readonly("cell_fall", &eda::TimingArc::get_cell_fall)
        .def_property_readonly("rise_transition",
                               &eda::TimingArc::get_rise_transition)
        .def_property_readonly("fall_transition",
                               &eda::TimingArc::get_fall_transition)
        .def_property_readonly("rise_constraint",
                               &eda::TimingArc::get_rise_constraint)
        .def_property_readonly("fall_constraint",
                               &eda::TimingArc::get_fall_constraint)
        .def("get_related_pin", &eda::TimingArc::get_related_pin)
        .def("get_timing_type", &eda::TimingArc::get_timing_type)
        .def("get_timing_sense", &eda::TimingArc::get_timing_sense)
        .def("get_cell_rise", &eda::TimingArc::get_cell_rise)
        .def("get_cell_fall", &eda::TimingArc::get_cell_fall)
        .def("get_rise_transition", &eda::TimingArc::get_rise_transition)
        .def("get_fall_transition", &eda::TimingArc::get_fall_transition)
        .def("get_rise_constraint", &eda::TimingArc::get_rise_constraint)
        .def("get_fall_constraint", &eda::TimingArc::get_fall_constraint);

    py::class_<eda::CellPin>(module, "CellPin", "A pin in a standard cell.")
        .def_property_readonly("name", &eda::CellPin::get_name)
        .def_property_readonly("direction", &eda::CellPin::get_direction)
        .def_property_readonly("function", &eda::CellPin::get_function)
        .def_property_readonly("capacitance", &eda::CellPin::get_capacitance)
        .def_property_readonly("timing_arcs", &eda::CellPin::get_timing_arcs)
        .def("get_name", &eda::CellPin::get_name)
        .def("get_direction", &eda::CellPin::get_direction)
        .def("get_function", &eda::CellPin::get_function)
        .def("get_capacitance", &eda::CellPin::get_capacitance)
        .def("get_timing_arcs", &eda::CellPin::get_timing_arcs);

    py::class_<eda::StandardCell>(module, "StandardCell",
                                  "A cell definition from a Liberty library.")
        .def_property_readonly("name", &eda::StandardCell::get_name)
        .def_property_readonly("area", &eda::StandardCell::get_area)
        .def_property_readonly("pins", &eda::StandardCell::get_pins)
        .def("get_name", &eda::StandardCell::get_name)
        .def("get_area", &eda::StandardCell::get_area)
        .def("get_pins", &eda::StandardCell::get_pins);

    py::class_<eda::StandardCellLibrary>(
        module, "StandardCellLibrary", "A parsed Liberty standard-cell library.")
        .def_property_readonly("name", &eda::StandardCellLibrary::get_name)
        .def_property_readonly("time_unit",
                               &eda::StandardCellLibrary::get_time_unit)
        .def_property_readonly("capacitance_unit",
                               &eda::StandardCellLibrary::get_capacitance_unit)
        .def_property_readonly("cells", &eda::StandardCellLibrary::get_cells)
        .def_property_readonly("cell_count",
                               &eda::StandardCellLibrary::get_cell_count)
        .def("get_name", &eda::StandardCellLibrary::get_name)
        .def("get_time_unit", &eda::StandardCellLibrary::get_time_unit)
        .def("get_capacitance_unit",
             &eda::StandardCellLibrary::get_capacitance_unit)
        .def("get_cells", &eda::StandardCellLibrary::get_cells)
        .def("get_cell_count", &eda::StandardCellLibrary::get_cell_count);

    module.def("parse_liberty", &eda::parse_liberty, py::arg("path"),
               "Parse cells, pins, Boolean functions, and timing tables from a Liberty file.");
    module.def("parse_liberty_source", &eda::parse_liberty_source,
               py::arg("source"),
               "Parse cells, pins, Boolean functions, and timing tables from Liberty source.");
    module.def("parse_verilog", &eda::parse_verilog, py::arg("path"),
               "Parse one Verilog module into an unmapped RTL representation.");
    module.def("parse_verilog_file", &eda::parse_verilog_file, py::arg("path"),
               "Parse one Verilog module using slang into an RTL Module.");
    module.def("generic_map",
               py::overload_cast<const eda::Module&>(&eda::generic_map),
               py::arg("rtl_module"),
               "Lower a parsed RTL module (assignments and always blocks) into "
               "technology-independent generic logic cells.");
    module.def("generic_map",
               py::overload_cast<const eda::Project&, const std::string&,
                                 const std::map<std::string, std::string>&>(
                   &eda::generic_map),
               py::arg("project"), py::arg("top") = "",
               py::arg("parameters") = std::map<std::string, std::string>{},
               "Lower the hierarchy below `top` of a parsed project into one flat "
               "generic netlist; `parameters` overrides top-level parameters.");
    module.def("tech_map",
               py::overload_cast<const eda::Module&, const eda::StandardCellLibrary&>(
                   &eda::tech_map),
               py::arg("generic_module"), py::arg("cell_library"),
               "Map generic logic cells to cells in the supplied Liberty library.");
    module.def("tech_map",
               py::overload_cast<const eda::Project&, const eda::StandardCellLibrary&,
                                 const std::string&,
                                 const std::map<std::string, std::string>&>(
                   &eda::tech_map),
               py::arg("project"), py::arg("cell_library"), py::arg("top") = "",
               py::arg("parameters") = std::map<std::string, std::string>{},
               "Generic-map and technology-map the hierarchy below `top` of a project.");
    module.def("fpga_map",
               py::overload_cast<const eda::Module&, std::size_t,
                                 const eda::StandardCellLibrary&>(&eda::fpga_map),
               py::arg("generic_module"), py::arg("lut_size") = 4,
               py::arg("library") = eda::get_default_fpga_library(),
               "Map a generic netlist onto K-input LUTs (K = 2..6); flops pass through. "
               "Instances carry cell data (area, pins, timing) from `library`.");
    module.def("fpga_map",
               py::overload_cast<const eda::Project&, const std::string&,
                                 const std::map<std::string, std::string>&, std::size_t,
                                 const eda::StandardCellLibrary&>(&eda::fpga_map),
               py::arg("project"), py::arg("top") = "",
               py::arg("parameters") = std::map<std::string, std::string>{},
               py::arg("lut_size") = 4,
               py::arg("library") = eda::get_default_fpga_library(),
               "Generic-map the hierarchy below `top` and map it onto K-input LUTs.");
    module.def("default_fpga_library", &eda::get_default_fpga_library,
               py::return_value_policy::copy,
               "Built-in generic FPGA cell model (LUT2-6, flops, latch, tribuf).");
    module.def("to_blif", &eda::to_blif, py::arg("module"),
               "Serialize a LUT netlist as BLIF.");
    module.def("write_blif", &eda::write_blif, py::arg("module"), py::arg("path"),
               "Write a LUT netlist to a BLIF file.");
    module.def("parse_verilog_project", &eda::parse_verilog_project,
               py::arg("directory"),
               "Parse and elaborate .v files recursively using slang (Verilog-2005).");
    module.def("to_json", &eda::to_json, py::arg("module"),
               "Serialize a Module using the versioned eda-netlist JSON format.");
    module.def("write_json", &eda::write_json, py::arg("module"), py::arg("path"),
               "Write a Module to the versioned eda-netlist JSON format.");
}
