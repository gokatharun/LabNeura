#include "labneura/eda/netlist.h"

#include "labneura/eda/native_verilog_project_context.h"
#include "labneura/eda/native_verilog_project_projection.h"

#include <algorithm>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace labneura::eda {
namespace {

[[noreturn]] void throw_native_diagnostics(
    const detail::NativeVerilogProjectContext& context,
    const std::string& operation) {
    std::string message = "native Verilog frontend failed to " + operation;
    for (const auto& diagnostic : context.parsed.diagnostics) {
        if (diagnostic.severity == VerilogSyntaxDiagnosticSeverity::Error)
            message += "\n[" + diagnostic.code + "] " + diagnostic.message;
    }
    if (context.elaboration) {
        for (const auto& diagnostic : context.elaboration->diagnostics)
            message += "\n[" + diagnostic.code + "] " + diagnostic.message;
    }
    if (context.types) {
        for (const auto& diagnostic : context.types->diagnostics)
            message += "\n[" + diagnostic.code + "] " + diagnostic.message;
    }
    throw std::runtime_error(message);
}

Project project_files(const std::vector<std::filesystem::path>& paths,
                      const std::string& operation) {
    auto context = detail::build_native_verilog_project_context_from_files(paths);
    if (!context->ready())
        throw_native_diagnostics(*context, operation);
    auto projection = detail::project_native_verilog(std::move(context));
    if (!projection.succeeded()) {
        std::string message = "native Verilog project projection failed";
        for (const auto& diagnostic : projection.diagnostics)
            message += "\n[" + diagnostic.code + "] " + diagnostic.message;
        throw std::runtime_error(message);
    }
    return std::move(projection.project);
}

} // namespace

Module parse_verilog(const std::string& path) {
    const auto project = project_files({std::filesystem::path(path)}, "parse source");
    std::vector<const Module*> roots;
    for (const auto& candidate : project.get_modules()) {
        bool instantiated = false;
        for (const auto& module : project.get_modules()) {
            for (const auto& instance : module.get_instances())
                instantiated |= instance.get_cell_type() == candidate.get_name();
        }
        if (!instantiated)
            roots.push_back(&candidate);
    }
    if (roots.size() != 1)
        throw std::runtime_error(
            "native parse_verilog requires exactly one top-level module");
    return *roots.front();
}

Module read_verilog(const std::string& path) {
    return parse_verilog(path);
}

Project parse_verilog_directory(const std::string& directory) {
    const std::filesystem::path root(directory);
    if (!std::filesystem::is_directory(root))
        throw std::runtime_error("Verilog project path is not a directory: " + directory);
    std::vector<std::filesystem::path> paths;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(root)) {
        if (!entry.is_regular_file())
            continue;
        const auto extension = entry.path().extension();
        if (extension == ".v" || extension == ".sv")
            paths.push_back(entry.path());
    }
    std::sort(paths.begin(), paths.end());
    if (paths.empty())
        throw std::runtime_error("no .v or .sv files found in project: " + directory);
    return project_files(paths, "parse project");
}

} // namespace labneura::eda
