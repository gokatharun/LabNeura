#include "labneura/eda/verilog_native_lowering_internal.h"

#include <algorithm>
#include <limits>
#include <unordered_set>

namespace labneura::eda::detail {
namespace {

std::optional<PortDirection> direction_for(VerilogPortDirection direction) {
    switch (direction) {
    case VerilogPortDirection::Input: return PortDirection::Input;
    case VerilogPortDirection::Output: return PortDirection::Output;
    case VerilogPortDirection::Inout: return PortDirection::InOut;
    default: return std::nullopt;
    }
}

const VerilogSemanticSymbol* module_symbol(
    const VerilogNativeLoweringSession& session) {
    if (!session.semantic.model.compilation_unit_scope) return nullptr;
    const auto& symbols = session.semantic.model.compilation_unit_scope->symbols_by_name;
    const auto found = symbols.find(session.module.name);
    return found == symbols.end() ? nullptr : found->second;
}

bool valid_dimensions(VerilogNativeLoweringSession& session,
                      const VerilogSemanticSymbol& symbol,
                      const VerilogIntegralType& type) {
    if (!type.is_known_integral() || type.category != VerilogTypeCategory::Integral ||
        type.unpacked_element_count || !type.unpacked_dimensions.empty() ||
        !symbol.unpacked_dimensions.empty()) {
        session.diagnose("UnsupportedNativeSignalType",
                         "native lowering supports only flat integral signals",
                         symbol.declaration_location);
        return false;
    }
    if (type.packed_dimensions.size() > 1 || symbol.packed_dimensions.size() > 1) {
        session.diagnose("UnsupportedNativePackedDimensions",
                         "native lowering supports at most one packed range",
                         symbol.declaration_location);
        return false;
    }
    if (type.packed_dimensions.size() == 1 &&
        (!type.packed_dimensions.front().left ||
         !type.packed_dimensions.front().right)) {
        session.diagnose("UnresolvedNativePackedRange",
                         "native lowering requires resolved packed range bounds",
                         symbol.declaration_location);
        return false;
    }
    return true;
}

bool add_signal(VerilogNativeLoweringSession& session,
                const VerilogSemanticSymbol* symbol,
                std::optional<PortDirection> direction) {
    if (!symbol || session.signal(symbol)) return true;
    const auto type_it = session.types.symbol_types.find(symbol);
    if (type_it == session.types.symbol_types.end() ||
        !valid_dimensions(session, *symbol,
                          type_it == session.types.symbol_types.end()
                              ? VerilogIntegralType{} : type_it->second))
        return false;
    const auto& type = type_it->second;
    if (direction == PortDirection::InOut) {
        session.diagnose("UnsupportedNativeInoutPort",
                         "native lowering does not support inout ports",
                         symbol->declaration_location);
        return false;
    }
    const size_t width = *type.bit_width;
    std::optional<int64_t> left, right;
    if (!type.packed_dimensions.empty()) {
        left = type.packed_dimensions.front().left;
        right = type.packed_dimensions.front().right;
    }
    VerilogNativeLoweringSignal lowered{symbol, type, symbol->name, {}, direction};
    lowered.bits.reserve(width);
    const auto net_kind = direction == PortDirection::Input ? NetType::Input :
                          direction == PortDirection::Output ? NetType::Output :
                                                               NetType::Internal;
    for (size_t bit = 0; bit < width; ++bit) {
        std::string net_name = symbol->name;
        if (left && right) {
            const int64_t delta = *left >= *right
                ? static_cast<int64_t>(bit) : -static_cast<int64_t>(bit);
            const int64_t index = *right + delta;
            net_name += "[" + std::to_string(index) + "]";
        }
        lowered.bits.push_back(net_name);
        session.add_net(Net(net_name, net_kind));
    }
    if (direction) {
        session.ports.emplace_back(symbol->name, *direction, width, left, right,
                                   std::nullopt, std::nullopt, width);
    }
    session.signal_indices.emplace(symbol, session.signals.size());
    session.signals.push_back(std::move(lowered));
    return true;
}

} // namespace

bool initialize_verilog_native_module(VerilogNativeLoweringSession& session) {
    const auto* module = module_symbol(session);
    if (!module || !module->member_scope) {
        session.diagnose("MissingNativeModuleScope",
                         "semantic module scope is unavailable",
                         session.module.location, std::nullopt, false);
        return false;
    }
    std::unordered_set<std::string> port_names;
    bool ok = true;
    for (const auto& port : session.module.ports) {
        port_names.insert(port.name);
        const auto found = module->member_scope->symbols_by_name.find(port.name);
        if (found == module->member_scope->symbols_by_name.end()) {
            session.diagnose("MissingNativePortSymbol",
                             "semantic symbol for port '" + port.name + "' is unavailable",
                             port.location, std::nullopt, false);
            ok = false;
            continue;
        }
        auto direction = direction_for(port.direction);
        if (!direction || port.direction == VerilogPortDirection::Ref) {
            session.diagnose("UnsupportedNativePortDirection",
                             "native lowering requires input or output ports",
                             port.location);
            ok = false;
            continue;
        }
        ok = add_signal(session, found->second, direction) && ok;
    }
    for (const auto& declaration : session.module.declarations) {
        if (port_names.contains(declaration.name)) continue;
        const auto found = module->member_scope->symbols_by_name.find(declaration.name);
        if (found == module->member_scope->symbols_by_name.end()) {
            session.diagnose("MissingNativeSignalSymbol",
                             "semantic symbol for signal '" + declaration.name + "' is unavailable",
                             declaration.location, std::nullopt, false);
            ok = false;
            continue;
        }
        const auto type_it = session.types.symbol_types.find(found->second);
        if (type_it != session.types.symbol_types.end() &&
            type_it->second.category == VerilogTypeCategory::Array) {
            const auto& array_type = type_it->second;
            const bool supported = declaration.unpacked_dimensions.size() == 1 &&
                array_type.unpacked_dimensions.size() == 1 &&
                array_type.unpacked_element_count &&
                *array_type.unpacked_element_count > 0 &&
                array_type.element_type &&
                array_type.element_type->is_known_integral() &&
                array_type.element_type->category == VerilogTypeCategory::Integral &&
                array_type.element_type->packed_dimensions.size() <= 1 &&
                array_type.unpacked_dimensions.front().left &&
                array_type.unpacked_dimensions.front().right &&
                *array_type.unpacked_dimensions.front().left == 0 &&
                *array_type.unpacked_dimensions.front().right >= 0;
            if (!supported ||
                static_cast<uint64_t>(*array_type.unpacked_dimensions.front().right) + 1 !=
                    *array_type.unpacked_element_count) {
                session.diagnose(
                    "UnsupportedNativeMemoryDeclaration",
                    "native memory inference requires one static [0:DEPTH-1] unpacked dimension and integral packed words",
                    declaration.location);
                ok = false;
                continue;
            }
            session.memory_declarations.emplace(
                found->second,
                VerilogNativeMemoryDeclaration{
                    found->second, &declaration, *array_type.element_type,
                    *array_type.unpacked_element_count});
            continue;
        }
        ok = add_signal(session, found->second, std::nullopt) && ok;
    }
    return ok;
}

bool lower_verilog_native_continuous_assignments(
    VerilogNativeLoweringSession& session) {
    bool ok = true;
    for (const auto& assignment : session.module.assignments) {
        if (!assignment.left || !assignment.right) {
            session.diagnose("InvalidNativeContinuousAssignment",
                             "continuous assignment is missing an operand",
                             assignment.location, std::nullopt, false);
            ok = false;
            continue;
        }
        auto target = session.resolve_lvalue(*assignment.left);
        for (const auto& diagnostic : target.diagnostics)
            session.diagnose(diagnostic.code, diagnostic.message, diagnostic.location,
                             diagnostic.related_location);
        auto value = target.target && target.target->is_integral
            ? lower_verilog_native_expression(
                  session, *assignment.right, {},
                  target.target->bit_width)
            : std::optional<VerilogNativeLoweringValue>{};
        if (!value || !target.target || target.has_errors()) {
            ok = false;
            continue;
        }
        const auto& resolved = *target.target;
        for (const auto& destination : resolved.destinations) {
            auto* signal = session.signal(destination.symbol);
            if (!signal) {
                session.diagnose("UnresolvedNativeAssignmentDestination",
                                 "continuous assignment destination is not a declared module signal",
                                 assignment.left->location);
                ok = false;
                continue;
            }
            for (size_t i = 0; i < destination.bit_positions.size(); ++i) {
                if (!destination.bit_positions[i]) continue;
                const size_t source_bit = destination.target_lsb_offset + i;
                const size_t target_bit = *destination.bit_positions[i];
                if (value->bits.empty() || target_bit >= signal->bits.size()) {
                    session.diagnose("InvalidNativeAssignmentBit",
                                     "continuous assignment bit mapping is out of range",
                                     assignment.left->location);
                    ok = false;
                    continue;
                }
                std::string source;
                if (source_bit < value->bits.size()) {
                    source = value->bits[source_bit];
                } else if (value->is_signed) {
                    source = value->bits.back();
                } else {
                    source = session.constant(false);
                }
                session.emit_buffer(source, signal->bits[target_bit]);
            }
        }
    }
    return ok && !session.unsupported && !session.invalid;
}

} // namespace labneura::eda::detail
