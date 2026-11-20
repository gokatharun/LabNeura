#include "labneura/eda/verilog_assignment.h"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>

namespace labneura::eda {
namespace {

void diagnose(VerilogAssignmentResult& result, std::string code,
              std::string message, const VerilogSyntaxLocation& location,
              std::optional<VerilogSyntaxLocation> related = {}) {
    result.diagnostics.push_back(
        {std::move(code), std::move(message), location, std::move(related)});
}

VerilogAssignmentDiagnostic diagnostic(
    std::string code, std::string message,
    const VerilogSyntaxLocation& location,
    std::optional<VerilogSyntaxLocation> related = {}) {
    return {std::move(code), std::move(message), location, std::move(related)};
}

} // namespace

VerilogAssignmentResult assign_verilog_lvalue(
    const VerilogResolvedAssignmentTarget& target,
    const VerilogConstantValue& value,
    VerilogConstantEnvironment& environment,
    VerilogAssignmentOptions options) {
    VerilogAssignmentResult result;
    const VerilogSyntaxLocation& location =
        target.location.source ? target.location : value.location;
    if (!target.is_integral || target.bit_width == 0) {
        diagnose(result, "InvalidAssignmentTargetWidth",
                 "assignment target width must be known, nonzero, and integral",
                 location);
        return result;
    }
    if (target.bit_width > options.maximum_width ||
        value.bit_width > options.maximum_width) {
        diagnose(result, "AssignmentWidthExceeded",
                 "assignment source or target exceeds the configured width limit",
                 location);
        return result;
    }
    if (value.bits.size() != value.bit_width) {
        diagnose(result, "MalformedAssignmentValue",
                 "assignment value bit storage does not match its declared width",
                 location);
        return result;
    }
    if (target.destinations.empty()) {
        diagnose(result, "EmptyAssignmentTarget",
                 "resolved assignment target contains no writable storage bits",
                 location);
        return result;
    }

    // Assignment conversion first sizes the RHS to the complete lvalue width.
    // Sign extension is controlled by the RHS expression; destination
    // signedness affects the stored type but not this conversion.
    std::vector<VerilogLogicBit> converted(target.bit_width, VerilogLogicBit::Zero);
    for (size_t bit = 0; bit < target.bit_width; ++bit) {
        if (value.unbased_unsized && !value.bits.empty())
            converted[bit] = value.bits.front();
        else if (bit < value.bit_width)
            converted[bit] = value.bits[bit];
        else if (value.is_signed && !value.bits.empty())
            converted[bit] = value.bits.back();
    }

    struct StagedValue {
        VerilogConstantValue value;
        std::unordered_set<size_t> written_positions;
    };
    std::unordered_map<std::string, StagedValue> staged;
    for (const auto& destination : target.destinations) {
        if (destination.storage_name.empty() ||
            !destination.declared_type.is_known_integral() ||
            !destination.declared_type.bit_width ||
            destination.bit_positions.empty()) {
            diagnose(result, "InvalidAssignmentDestination",
                     "assignment destination has an invalid storage name or type",
                     location,
                     destination.symbol
                         ? std::optional<VerilogSyntaxLocation>(
                               destination.symbol->declaration_location)
                         : std::nullopt);
            return result;
        }
        if (destination.target_lsb_offset > target.bit_width ||
            destination.bit_positions.size() >
                target.bit_width - destination.target_lsb_offset) {
            diagnose(result, "AssignmentDestinationOutOfRange",
                     "assignment destination bit range exceeds the lvalue width",
                     location);
            return result;
        }

        auto staged_entry = staged.find(destination.storage_name);
        if (staged_entry == staged.end()) {
            const auto stored = environment.find(destination.storage_name);
            if (stored == environment.end()) {
                diagnose(result, "MissingAssignmentStorage",
                         "assignment storage '" + destination.storage_name +
                             "' is absent from the active execution environment",
                         location,
                         destination.symbol
                             ? std::optional<VerilogSyntaxLocation>(
                                   destination.symbol->declaration_location)
                             : std::nullopt);
                return result;
            }
            if (stored->second.bit_width != *destination.declared_type.bit_width ||
                stored->second.bits.size() != stored->second.bit_width) {
                diagnose(result, "AssignmentStorageTypeMismatch",
                         "assignment storage width does not match its declared type",
                         location,
                         destination.symbol
                             ? std::optional<VerilogSyntaxLocation>(
                                   destination.symbol->declaration_location)
                             : std::nullopt);
                return result;
            }
            VerilogConstantValue typed_storage = stored->second;
            typed_storage.is_signed = destination.declared_type.is_signed;
            staged_entry = staged.emplace(
                destination.storage_name,
                StagedValue{std::move(typed_storage), {}}).first;
        }

        for (size_t bit = 0; bit < destination.bit_positions.size(); ++bit) {
            const auto position = destination.bit_positions[bit];
            // An out-of-range select is a legal null destination and consumes
            // its corresponding RHS bit without changing storage.
            if (!position)
                continue;
            if (*position >= staged_entry->second.value.bit_width) {
                diagnose(result, "AssignmentBitOutOfRange",
                         "resolved assignment bit exceeds destination storage width",
                         location);
                return result;
            }
            if (!staged_entry->second.written_positions.insert(*position).second) {
                diagnose(result, "OverlappingAssignmentDestinations",
                         "assignment destinations overlap the same storage bit",
                         location,
                         destination.symbol
                             ? std::optional<VerilogSyntaxLocation>(
                                   destination.symbol->declaration_location)
                             : std::nullopt);
                return result;
            }
            VerilogLogicBit assigned =
                converted[destination.target_lsb_offset + bit];
            const bool destination_four_state = target.destinations.size() == 1
                                                    ? target.four_state
                                                    : destination.declared_type.four_state;
            if (!destination_four_state &&
                (assigned == VerilogLogicBit::X || assigned == VerilogLogicBit::Z))
                assigned = VerilogLogicBit::Zero;
            staged_entry->second.value.bits[*position] = assigned;
        }
    }

    for (auto& [name, staged_value] : staged)
        environment[name] = std::move(staged_value.value);
    result.assigned = true;
    return result;
}

VerilogLvalueReadResult read_verilog_lvalue(
    const VerilogResolvedAssignmentTarget& target,
    const VerilogConstantEnvironment& environment, size_t maximum_width) {
    VerilogLvalueReadResult result;
    if (!target.is_integral || target.bit_width == 0 ||
        target.bit_width > maximum_width) {
        result.diagnostics.push_back(diagnostic(
            "InvalidAssignmentTargetWidth",
            "lvalue read width must be known, nonzero, integral, and within the configured limit",
            {}));
        return result;
    }
    VerilogConstantValue read_value;
    read_value.bit_width = target.bit_width;
    read_value.is_signed = target.is_signed;
    read_value.bits.assign(target.bit_width, VerilogLogicBit::X);
    if (target.destinations.empty()) {
        result.diagnostics.push_back(diagnostic(
            "EmptyAssignmentTarget", "resolved lvalue contains no destinations",
            target.location));
        return result;
    }
    std::vector<bool> mapped_bits(target.bit_width, false);
    for (const auto& destination : target.destinations) {
        const auto stored = environment.find(destination.storage_name);
        if (stored == environment.end()) {
            result.diagnostics.push_back(diagnostic(
                "MissingAssignmentStorage",
                "assignment storage '" + destination.storage_name +
                    "' is absent from the active execution environment",
                destination.symbol ? destination.symbol->declaration_location
                                   : VerilogSyntaxLocation{},
                destination.symbol
                    ? std::optional<VerilogSyntaxLocation>(
                          destination.symbol->declaration_location)
                    : std::nullopt));
            return result;
        }
        if (stored->second.bits.size() != stored->second.bit_width) {
            result.diagnostics.push_back(diagnostic(
                "MalformedAssignmentStorage",
                "assignment storage bit count does not match its declared width",
                destination.symbol ? destination.symbol->declaration_location
                                   : VerilogSyntaxLocation{}));
            return result;
        }
        if (!destination.declared_type.bit_width ||
            !destination.declared_type.is_known_integral() ||
            stored->second.bit_width != *destination.declared_type.bit_width) {
            result.diagnostics.push_back(diagnostic(
                "AssignmentStorageTypeMismatch",
                "assignment storage width does not match its resolved declared type",
                target.location,
                destination.symbol
                    ? std::optional<VerilogSyntaxLocation>(
                          destination.symbol->declaration_location)
                    : std::nullopt));
            return result;
        }
        if (destination.target_lsb_offset > target.bit_width ||
            destination.bit_positions.size() >
                target.bit_width - destination.target_lsb_offset) {
            result.diagnostics.push_back(diagnostic(
                "AssignmentDestinationOutOfRange",
                "assignment destination bit range exceeds the lvalue width",
                target.location));
            return result;
        }
        for (size_t bit = 0; bit < destination.bit_positions.size(); ++bit) {
            const size_t target_bit = destination.target_lsb_offset + bit;
            if (mapped_bits[target_bit]) {
                result.diagnostics.push_back(diagnostic(
                    "OverlappingAssignmentDestinations",
                    "resolved lvalue maps multiple source bits to one target bit",
                    target.location));
                return result;
            }
            mapped_bits[target_bit] = true;
            const auto position = destination.bit_positions[bit];
            if (position && *position >= stored->second.bit_width) {
                result.diagnostics.push_back(diagnostic(
                    "AssignmentBitOutOfRange",
                    "resolved lvalue bit position exceeds destination storage width",
                    target.location));
                return result;
            }
            if (position)
                read_value.bits[target_bit] = stored->second.bits[*position];
            // Only null mappings represent legal out-of-range select bits.
        }
    }
    result.value = std::move(read_value);
    return result;
}

} // namespace labneura::eda
