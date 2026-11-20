#include "labneura/eda/netlist.h"
#include "labneura/eda/verilog_elaboration.h"
#include "labneura/eda/native_verilog_project_context.h"
#include "labneura/eda/native_verilog_project_projection.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
using namespace labneura::eda;
using namespace labneura::eda::detail;

void require(bool condition, std::string_view message) {
    if (!condition)
        throw std::runtime_error(std::string(message));
}

class TemporaryDirectory {
public:
    TemporaryDirectory() {
        const auto nonce =
            std::chrono::steady_clock::now().time_since_epoch().count();
        path_ = std::filesystem::temp_directory_path() /
                ("labneura-native-project-" + std::to_string(nonce));
        std::filesystem::create_directories(path_);
    }

    ~TemporaryDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }

    const std::filesystem::path& path() const noexcept { return path_; }

    std::filesystem::path write(std::string_view name, std::string_view text) {
        const auto path = path_ / name;
        std::ofstream output(path);
        if (!output)
            throw std::runtime_error("could not create projection test source");
        output << text;
        return path;
    }

    std::string read(const std::filesystem::path& path) {
        std::ifstream input(path);
        if (!input)
            throw std::runtime_error("could not read projection test source");
        return {std::istreambuf_iterator<char>(input), {}};
    }

private:
    std::filesystem::path path_;
};

void compare_module_summary(const Module& native, const Module& slang) {
    require(native.get_name() == slang.get_name(),
            "native and Slang module names differ: native='" +
                native.get_name() + "', Slang='" + slang.get_name() + "'");
    require(native.get_ports().size() == slang.get_ports().size(),
            "native and Slang port counts differ");
    for (size_t index = 0; index < native.get_ports().size(); ++index) {
        const auto& left = native.get_ports()[index];
        const auto& right = slang.get_ports()[index];
        require(left.get_name() == right.get_name() &&
                    left.get_direction() == right.get_direction() &&
                    left.get_width() == right.get_width() &&
                    left.get_left_index() == right.get_left_index() &&
                    left.get_right_index() == right.get_right_index(),
                "native and Slang port projections differ");
    }
    require(native.get_nets().size() == slang.get_nets().size(),
            "native and Slang net counts differ");
    for (size_t index = 0; index < native.get_nets().size(); ++index) {
        const auto& left = native.get_nets()[index];
        const auto& right = slang.get_nets()[index];
        require(left.get_name() == right.get_name() &&
                    left.get_kind() == right.get_kind() &&
                    left.get_width() == right.get_width() &&
                    left.get_left_index() == right.get_left_index() &&
                    left.get_right_index() == right.get_right_index(),
                "native and Slang net projections differ");
    }
    require(native.get_instances().size() == slang.get_instances().size(),
            "native and Slang instance counts differ");
    for (size_t index = 0; index < native.get_instances().size(); ++index) {
        const auto& left = native.get_instances()[index];
        const auto& right = slang.get_instances()[index];
        require(left.get_name() == right.get_name() &&
                    left.get_cell_type() == right.get_cell_type() &&
                    left.get_connections().size() ==
                        right.get_connections().size(),
                "native and Slang instance summaries differ");
        for (size_t connection = 0;
             connection < left.get_connections().size(); ++connection) {
            require(left.get_connections()[connection].get_pin() ==
                            right.get_connections()[connection].get_pin() &&
                        left.get_connections()[connection].get_net() ==
                            right.get_connections()[connection].get_net(),
                    "native and Slang instance connections differ");
        }
    }
}

void test_multiple_sources_projection_and_slang_parity() {
    TemporaryDirectory sources;
    const auto leaf_path = sources.write(
        "a_leaf.sv",
        "module leaf #(parameter int WIDTH = 4) "
        "(input logic [WIDTH-1:0] a, output logic y);\n"
        "  wire [3:0] local_wire;\n"
        "  assign y = a[0];\n"
        "endmodule\n");
    const auto top_path = sources.write(
        "b_top.sv",
        "module top #(parameter int WIDTH = 8) "
        "(input logic [WIDTH-1:0] data, output logic y);\n"
        "  wire [WIDTH-1:0] internal;\n"
        "  assign internal = data;\n"
        "  leaf #(.WIDTH(3)) u_leaf(.a(data[2:0]), .y(y));\n"
        "endmodule\n");
    sources.write("c_spare.sv",
                  "module spare(input logic unused); endmodule\n");

    VerilogElaborationOptions options;
    options.top_module = "top";
    auto context = build_native_verilog_project_context_from_files(
        {leaf_path, top_path, sources.path() / "c_spare.sv"}, {}, {}, options);
    require(context->ready(),
            "valid multi-source native project context was not ready");
    auto projection = project_native_verilog(context);
    require(projection.succeeded(),
            "native Project/Module projection reported diagnostics");
    require(projection.project.get_source_files().size() == 3 &&
                projection.project.get_module_count() == 3,
            "native projection lost source files or module definitions");
    require(projection.project.get_modules()[0].get_name() == "spare" &&
                projection.project.get_modules()[1].get_name() == "top" &&
                projection.project.get_modules()[2].get_name() == "leaf",
            "native module order differs from elaborated project order");

    const auto slang = parse_verilog_directory(sources.path().string());
    require(slang.get_module_count() == projection.project.get_module_count(),
            "native and Slang project module counts differ");
    for (size_t index = 0; index < slang.get_module_count(); ++index)
        compare_module_summary(projection.project.get_modules()[index],
                               slang.get_modules()[index]);

    const auto& top = projection.project.get_modules()[1];
    const auto combined_path = sources.write(
        "combined.sv", sources.read(leaf_path) + sources.read(top_path));
    const auto slang_top = parse_verilog(combined_path.string());
    compare_module_summary(top, slang_top);
    require(top.get_child_modules().size() == 1 &&
                top.get_child_modules().front().get_hierarchy_instance_name() ==
                    "u_leaf" &&
                top.get_child_modules().front().get_ports().front().get_width() ==
                    3 &&
                slang_top.get_child_modules().size() == 1 &&
                slang_top.get_child_modules().front()
                        .get_hierarchy_instance_name() == "u_leaf" &&
                slang_top.get_child_modules().front().get_ports().front()
                        .get_width() == 3,
            "native hierarchy did not retain the instance parameter specialization");
    compare_module_summary(top.get_child_modules().front(),
                           slang_top.get_child_modules().front());
    require(top.get_instances().front().get_parameters().at("WIDTH") == "3",
            "native instance projection lost its explicit parameter override");
    require(top.get_assignments().size() == 1 &&
                top.get_assignments().front().get_lhs() == "internal",
            "native projection did not retain continuous assignments");
    require(projection.project.get_ast_json().empty() &&
                !projection.compatibility_gaps.empty(),
            "missing native AST JSON support was not identified");
    bool generic_mapping_rejected = false;
    try {
        (void)generic_map(projection.project, "top", {{"WIDTH", "9"}});
    } catch (const std::runtime_error& error) {
        generic_mapping_rejected =
            std::string_view(error.what()).find("native Project projections") !=
            std::string_view::npos;
    }
    require(generic_mapping_rejected,
            "native hierarchy was silently sent to the Slang behavioral mapper");

    std::weak_ptr<NativeVerilogProjectContext> retained = context;
    context.reset();
    require(!retained.expired(),
            "project did not retain its native context safely (owners=" +
                std::to_string(retained.use_count()) + ")");
    require(projection.project.get_module_count() == 3,
            "project became unusable after releasing the caller's context");
}

void test_top_parameter_overrides_change_projected_widths() {
    VerilogElaborationOptions options;
    options.top_module = "top";
    options.top_parameter_overrides.emplace("WIDTH", "10");
    auto context = build_native_verilog_project_context(
        {{"module top #(parameter int WIDTH = 4) "
          "(input logic [WIDTH-1:0] data); endmodule\n",
          "top.sv"}},
        {}, {}, options);
    require(context->ready(),
            "valid top-level parameter override did not elaborate");
    auto projection = project_native_verilog(context);
    require(projection.succeeded() &&
                projection.project.get_modules().front()
                        .get_ports()
                        .front()
                        .get_width() == 10 &&
                context->elaboration->top->parameters.at("WIDTH")
                        .to_uint64() == std::optional<uint64_t>(10),
            "top-level override did not specialize the native Project module");
}

void test_generated_hierarchy_and_connections() {
    auto context = build_native_verilog_project_context(
        {{"module leaf #(parameter int WIDTH = 1) "
          "(input logic [WIDTH-1:0] a); endmodule\n"
          "module top #(parameter int COUNT = 2) "
          "(input logic [COUNT-1:0] bus);\n"
          "  generate for (genvar i = 0; i < COUNT; i++) begin : lanes\n"
          "    logic scoped;\n"
          "    assign scoped = bus[i];\n"
          "    leaf #(.WIDTH(1)) u(.a(scoped));\n"
          "  end endgenerate\n"
          "endmodule\n",
          "generated.sv"}},
        {}, {}, VerilogElaborationOptions{.top_module = "top"});
    require(context->ready(),
            "supported generated hierarchy did not elaborate");
    auto projection = project_native_verilog(context);
    require(projection.succeeded(),
            "generated hierarchy projection reported diagnostics");
    const auto& top = projection.project.get_modules().front();
    require(top.get_instances().size() == 2 &&
                top.get_child_modules().size() == 2,
            "generated module instances were not projected");
    require(top.get_instances()[0].get_name() == "lanes[0].u" &&
                top.get_instances()[1].get_name() == "lanes[1].u" &&
                top.get_nets()[1].get_name() == "lanes[0].scoped" &&
                top.get_nets()[2].get_name() == "lanes[1].scoped" &&
                top.get_assignments()[0].get_lhs() == "lanes[0].scoped" &&
                top.get_assignments()[1].get_lhs() == "lanes[1].scoped" &&
                top.get_instances()[0].get_connections()[0].get_net() ==
                    "lanes[0].scoped" &&
                top.get_instances()[1].get_connections()[0].get_net() ==
                    "lanes[1].scoped",
            "generated scope names or parameterized connections differ");
}

void test_behavioral_projection_is_not_sent_to_generic_synthesis() {
    auto context = build_native_verilog_project_context(
        {{"module top(input logic a, output logic y);\n"
          "  always_comb y = a;\n"
          "endmodule\n",
          "behavioral.sv"}},
        {}, {}, VerilogElaborationOptions{.top_module = "top"});
    require(context->ready(),
            "supported behavioral module did not produce a native context");
    auto projection = project_native_verilog(context);
    require(projection.succeeded() &&
                projection.project.get_modules().front().is_behavioral(),
            "native projection did not preserve the behavioral marker");
    bool rejected = false;
    try {
        (void)generic_map(projection.project, "top");
    } catch (const std::runtime_error& error) {
        rejected = std::string_view(error.what()).find(
                       "behavioral native Project projections") !=
                   std::string_view::npos;
    }
    require(rejected,
            "behavioral native RTL was silently treated as a gate-level netlist");
}

} // namespace

int main() {
    try {
        test_multiple_sources_projection_and_slang_parity();
        test_top_parameter_overrides_change_projected_widths();
        test_generated_hierarchy_and_connections();
        test_behavioral_projection_is_not_sent_to_generic_synthesis();
    } catch (const std::exception& error) {
        std::cerr << "native project projection test failed: " << error.what()
                  << '\n';
        return 1;
    }
    return 0;
}
