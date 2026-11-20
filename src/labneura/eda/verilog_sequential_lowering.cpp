#include "verilog_native_lowering_internal.h"

#include <algorithm>
#include <map>
#include <optional>
#include <unordered_map>

namespace labneura::eda::detail {
namespace {

using Symbol = VerilogSemanticSymbol;
using Value = VerilogNativeLoweringValue;

const Symbol* symbol_of(const VerilogExpression* expression,
                        const VerilogNativeLoweringSession& session) {
    return expression ? session.resolve_symbol(*expression) : nullptr;
}

std::optional<bool> literal_bit(const VerilogExpression* expression,
                                const VerilogNativeLoweringSession& session) {
    if (!expression)
        return std::nullopt;
    const auto value = session.evaluate_constant(*expression);
    if (!value || value->bits.empty() || !value->is_fully_known())
        return std::nullopt;
    return value->bits.front() == VerilogLogicBit::One;
}

// Returns the signal value for which an if-condition is true. A false return
// means the condition is true while the signal is low.
std::optional<bool> condition_true_when_high(
    const VerilogExpression* expression, const Symbol* signal,
    const VerilogNativeLoweringSession& session) {
    if (!expression || !signal)
        return std::nullopt;
    if (expression->kind == VerilogExpressionKind::Identifier)
        return symbol_of(expression, session) == signal
                   ? std::optional<bool>{true}
                   : std::nullopt;
    if (expression->kind == VerilogExpressionKind::Unary && expression->first &&
        (expression->text == "!" || expression->text == "~")) {
        const auto nested = condition_true_when_high(expression->first.get(), signal,
                                                     session);
        return nested ? std::optional<bool>{!*nested} : std::nullopt;
    }
    if (expression->kind != VerilogExpressionKind::Binary ||
        !expression->first || !expression->second)
        return std::nullopt;

    const Symbol* lhs_symbol = symbol_of(expression->first.get(), session);
    const Symbol* rhs_symbol = symbol_of(expression->second.get(), session);
    const VerilogExpression* constant_expression = nullptr;
    bool signal_on_lhs = false;
    if (lhs_symbol == signal && !rhs_symbol) {
        constant_expression = expression->second.get();
        signal_on_lhs = true;
    } else if (rhs_symbol == signal && !lhs_symbol) {
        constant_expression = expression->first.get();
    } else {
        return std::nullopt;
    }
    const auto constant = literal_bit(constant_expression, session);
    if (!constant)
        return std::nullopt;

    bool equality_operator = expression->text == "==" || expression->text == "===";
    bool inequality_operator = expression->text == "!=" || expression->text == "!==";
    if (!equality_operator && !inequality_operator)
        return std::nullopt;
    (void)signal_on_lhs; // Equality and inequality are symmetric.
    const bool signal_value_when_true = equality_operator ? *constant : !*constant;
    return signal_value_when_true;
}

struct ResetLiteralValues {
    std::map<const Symbol*, std::vector<std::optional<bool>>> bits;
    bool found_assignment = false;
    bool valid = true;
};

bool collect_reset_assignments(const VerilogStatement* statement,
                               VerilogNativeLoweringSession& session,
                               ResetLiteralValues& reset) {
    if (!statement)
        return true;
    switch (statement->kind) {
        case VerilogStatementKind::Empty:
            return true;
        case VerilogStatementKind::Block:
            for (const auto& child : statement->statements)
                if (!collect_reset_assignments(child.get(), session, reset))
                    return false;
            return true;
        case VerilogStatementKind::ProceduralAssignment: {
            if (statement->assignment_kind != VerilogAssignmentKind::Nonblocking ||
                !statement->left || !statement->right) {
                reset.valid = false;
                return false;
            }
            const auto constant = session.evaluate_constant(*statement->right);
            const auto lvalue = session.resolve_lvalue(*statement->left);
            if (!constant || !constant->is_fully_known() || !lvalue.target) {
                reset.valid = false;
                return false;
            }
            reset.found_assignment = true;
            const auto& target = *lvalue.target;
            for (const auto& destination : target.destinations) {
                const auto* signal = session.signal(destination.symbol);
                if (!signal) {
                    reset.valid = false;
                    return false;
                }
                auto& destination_bits = reset.bits[destination.symbol];
                if (destination_bits.empty())
                    destination_bits.resize(signal->bits.size());
                const bool extend_sign = constant->is_signed &&
                                         !constant->bits.empty() &&
                                         constant->bits.back() == VerilogLogicBit::One;
                for (size_t bit = 0; bit < destination.bit_positions.size(); ++bit) {
                    const auto physical = destination.bit_positions[bit];
                    if (!physical)
                        continue;
                    const size_t source_bit = destination.target_lsb_offset + bit;
                    const VerilogLogicBit source_value =
                        source_bit < constant->bits.size()
                            ? constant->bits[source_bit]
                            : extend_sign ? VerilogLogicBit::One
                                          : VerilogLogicBit::Zero;
                    if (source_value != VerilogLogicBit::Zero &&
                        source_value != VerilogLogicBit::One) {
                        reset.valid = false;
                        return false;
                    }
                    auto& destination_bit = destination_bits[*physical];
                    if (destination_bit) {
                        reset.valid = false;
                        return false;
                    }
                    destination_bit = source_value == VerilogLogicBit::One;
                }
            }
            return true;
        }
        default:
            // A reset arm containing a nested branch, case, task, timing
            // control or loop is outside this canonical sequential subset.
            reset.valid = false;
            return false;
    }
}

bool find_reset_values(const VerilogStatement* statement,
                       const VerilogSequentialReset& reset_info,
                       VerilogNativeLoweringSession& session,
                       ResetLiteralValues& result) {
    if (!statement)
        return false;
    if (statement->kind == VerilogStatementKind::If && statement->expression) {
        const auto polarity = condition_true_when_high(statement->expression.get(),
                                                       reset_info.symbol, session);
        if (polarity) {
            const bool reset_when_condition_true =
                *polarity == reset_info.active_high;
            const auto* reset_arm = reset_when_condition_true
                                        ? statement->first_statement.get()
                                        : statement->second_statement.get();
            ResetLiteralValues candidate;
            if (collect_reset_assignments(reset_arm, session, candidate) &&
                candidate.valid && candidate.found_assignment) {
                result = std::move(candidate);
                return true;
            }
            return false;
        }
    }
    if (statement->kind == VerilogStatementKind::Block) {
        for (const auto& child : statement->statements)
            if (find_reset_values(child.get(), reset_info, session, result))
                return true;
    }
    return false;
}

std::optional<std::string> truth_bit(
    VerilogNativeLoweringSession& session, Value value,
    const VerilogSyntaxLocation& location) {
    if (value.bits.empty()) {
        session.diagnose("EmptySequentialCondition",
                         "sequential condition has no value bits", location);
        return std::nullopt;
    }
    std::string result = value.bits.front();
    for (size_t bit = 1; bit < value.bits.size(); ++bit)
        result = session.emit_binary("OR2", result, value.bits[bit]);
    return result;
}

std::optional<std::string> combine_guard(
    VerilogNativeLoweringSession& session, const std::string& outer,
    const std::string& inner) {
    const auto zero = session.constant(false);
    const auto one = session.constant(true);
    if (outer == one) return inner;
    if (inner == one) return outer;
    if (outer == zero || inner == zero) return zero;
    return session.emit_binary("AND2", outer, inner);
}

using NextState =
    std::unordered_map<const Symbol*, std::vector<std::string>>;

bool lower_nba_statement(VerilogNativeLoweringSession& session,
                         const VerilogStatement* statement,
                         const std::string& guard,
                         NextState& next_state) {
    if (!statement)
        return true;
    switch (statement->kind) {
        case VerilogStatementKind::Empty:
            return true;
        case VerilogStatementKind::Block:
            for (const auto& child : statement->statements)
                if (!lower_nba_statement(session, child.get(), guard, next_state))
                    return false;
            return true;
        case VerilogStatementKind::If: {
            if (!statement->expression) {
                session.diagnose("MissingSequentialCondition",
                                 "clocked if statement has no condition",
                                 statement->location);
                return false;
            }
            const auto condition = lower_verilog_native_expression(
                session, *statement->expression);
            if (!condition)
                return false;
            const auto condition_value = truth_bit(session, *condition,
                                                   statement->expression->location);
            if (!condition_value)
                return false;
            const auto false_condition = session.emit_not(*condition_value);
            const auto true_guard = combine_guard(session, guard, *condition_value);
            const auto false_guard = combine_guard(session, guard, false_condition);
            if (!true_guard || !false_guard)
                return false;
            if (!lower_nba_statement(session, statement->first_statement.get(),
                                     *true_guard, next_state))
                return false;
            return lower_nba_statement(session, statement->second_statement.get(),
                                       *false_guard, next_state);
        }
        case VerilogStatementKind::ProceduralAssignment: {
            if (statement->assignment_kind != VerilogAssignmentKind::Nonblocking ||
                !statement->left || !statement->right) {
                session.diagnose("UnsupportedSequentialAssignment",
                                 "native sequential lowering accepts only complete nonblocking assignments",
                                 statement->location);
                return false;
            }
            // Every NBA RHS reads the pre-edge signal values. The pending
            // next_state map is intentionally not used as an expression
            // environment, while update order still gives later NBAs priority.
            const auto lvalue = session.resolve_lvalue(*statement->left);
            const auto rhs = lvalue.target && lvalue.target->is_integral
                ? lower_verilog_native_expression(
                      session, *statement->right, {},
                      lvalue.target->bit_width)
                : std::optional<VerilogNativeLoweringValue>{};
            if (!rhs || !lvalue.target) {
                session.diagnose("UnsupportedSequentialLvalue",
                                 "sequential assignment lvalue could not be lowered statically",
                                 statement->left->location);
                return false;
            }
            const auto& target = *lvalue.target;
            auto resized = *rhs;
            const bool sign_extend = resized.is_signed && !resized.bits.empty();
            const auto extend_bit = sign_extend
                ? resized.bits.back()
                : session.constant(false);
            if (resized.bits.size() > target.bit_width)
                resized.bits.resize(target.bit_width);
            else
                resized.bits.resize(target.bit_width, extend_bit);

            for (const auto& destination : target.destinations) {
                const auto* signal = session.signal(destination.symbol);
                auto found = next_state.find(destination.symbol);
                if (!signal || found == next_state.end()) {
                    session.diagnose("UnsupportedSequentialDestination",
                                     "sequential destination is not a declared module signal",
                                     statement->left->location);
                    return false;
                }
                for (size_t bit = 0; bit < destination.bit_positions.size(); ++bit) {
                    const auto physical = destination.bit_positions[bit];
                    if (!physical)
                        continue; // Legal static out-of-range lvalue bit.
                    const size_t source_bit = destination.target_lsb_offset + bit;
                    if (*physical >= found->second.size() ||
                        source_bit >= resized.bits.size()) {
                        session.diagnose("SequentialLvalueRangeMismatch",
                                         "resolved assignment bit is outside its lowered signal or value width",
                                         statement->left->location);
                        return false;
                    }
                    const auto one = session.constant(true);
                    const auto zero = session.constant(false);
                    if (guard == one)
                        found->second[*physical] = resized.bits[source_bit];
                    else if (guard != zero)
                        found->second[*physical] = session.emit_mux(
                            found->second[*physical], resized.bits[source_bit], guard);
                }
            }
            return true;
        }
        default:
            session.diagnose("UnsupportedSequentialStatement",
                             "clocked process statement is outside the canonical sequential subset",
                             statement->location);
            return false;
    }
}

} // namespace

bool lower_verilog_native_sequential_process(
    VerilogNativeLoweringSession& session,
    const VerilogProcessAnalysis& analysis) {
    const auto& process = analysis.process;
    if (!process.procedural_block ||
        (process.kind != VerilogProcessKind::AlwaysFF &&
         process.kind != VerilogProcessKind::Always) ||
        analysis.synthesis_eligibility != VerilogSynthesisEligibility::Eligible ||
        analysis.sequential.knowledge != VerilogAnalysisKnowledge::Known ||
        analysis.sequential.clocks.size() != 1 ||
        analysis.sequential.has_ambiguous_clock_or_reset) {
        session.diagnose("UnsupportedSequentialProcess",
                         "native sequential lowering requires an eligible single-clock procedural process",
                         process.location);
        return false;
    }
    if (analysis.sequential.has_blocking_assignments ||
        !analysis.sequential.has_nonblocking_assignments) {
        session.diagnose("UnsupportedSequentialAssignmentMix",
                         "native sequential lowering requires nonblocking assignments without blocking assignments",
                         process.location);
        return false;
    }
    if (process.contains_timing_control || process.contains_unsupported_statement ||
        analysis.dependencies.knowledge != VerilogAnalysisKnowledge::Known ||
        analysis.dependencies.unresolved_callable_side_effects) {
        session.diagnose("UnsupportedSequentialDependency",
                         "clocked process contains unsupported timing, statements, or unresolved dependencies",
                         process.location);
        return false;
    }

    const auto& clock = analysis.sequential.clocks.front();
    const auto* clock_signal = session.signal(clock.symbol);
    if (!clock_signal || clock_signal->bits.size() != 1) {
        session.diagnose("UnsupportedClockSignal",
                         "native sequential lowering requires a declared one-bit clock signal",
                         clock.location);
        return false;
    }
    std::string clock_net = clock_signal->bits.front();
    if (clock.edge == VerilogEventEdge::Negedge)
        clock_net = session.emit_not(clock_net);
    else if (clock.edge != VerilogEventEdge::Posedge) {
        session.diagnose("UnsupportedClockEdge",
                         "native sequential lowering supports only edge-triggered clocks",
                         clock.location);
        return false;
    }

    NextState next_state;
    std::vector<const Symbol*> written_signals;
    for (const auto* symbol : analysis.dependencies.writes) {
        const auto* signal = session.signal(symbol);
        if (!signal) {
            session.diagnose("UnsupportedSequentialDestination",
                             "clocked process writes a signal outside the module netlist subset",
                             process.location);
            return false;
        }
        if (!next_state.contains(symbol)) {
            next_state.emplace(symbol, signal->bits);
            written_signals.push_back(symbol);
        }
    }
    if (next_state.empty()) {
        session.diagnose("SequentialProcessHasNoTargets",
                         "clocked process has no resolved register destinations",
                         process.location);
        return false;
    }

    for (const auto& reset : analysis.sequential.resets) {
        if (reset.synchronous)
            continue; // Synchronous reset stays as a data-path mux.
        if (!reset.asynchronous || !reset.reset_value_is_constant ||
            session.signal(reset.symbol) == nullptr) {
            session.diagnose("UnsupportedAsynchronousReset",
                             "asynchronous reset must be a resolved signal with a constant reset assignment",
                             reset.condition_location);
            return false;
        }
        ResetLiteralValues reset_values;
        if (!find_reset_values(process.procedural_block->body.get(), reset,
                               session, reset_values) ||
            !reset_values.valid || !reset_values.found_assignment) {
            session.diagnose("UnsupportedAsynchronousResetValue",
                             "reset branch must consist of static nonblocking assignments of known constants",
                             reset.condition_location);
            return false;
        }
        for (const auto& [symbol, values] : reset_values.bits) {
            const auto state = next_state.find(symbol);
            if (state == next_state.end() || values.size() != state->second.size() ||
                std::any_of(values.begin(), values.end(),
                            [](const auto& value) { return !value.has_value(); })) {
                session.diagnose("IncompleteAsynchronousResetValue",
                                 "every reset register bit must receive a known constant value",
                                 reset.condition_location);
                return false;
            }
        }
    }
    const size_t async_reset_count = static_cast<size_t>(std::count_if(
        analysis.sequential.resets.begin(), analysis.sequential.resets.end(),
        [](const auto& reset) { return reset.asynchronous; }));
    if (async_reset_count > 1) {
        session.diagnose("MultipleAsynchronousResets",
                         "native sequential lowering supports at most one asynchronous reset signal per process",
                         process.location);
        return false;
    }

    const auto one = session.constant(true);
    if (!lower_nba_statement(session, process.procedural_block->body.get(), one,
                             next_state))
        return false;

    const VerilogSequentialReset* async_reset = nullptr;
    for (const auto& reset : analysis.sequential.resets)
        if (reset.asynchronous) async_reset = &reset;
    std::string reset_active_low;
    if (async_reset) {
        const auto* reset_signal = session.signal(async_reset->symbol);
        if (!reset_signal || reset_signal->bits.size() != 1) {
            session.diagnose("UnsupportedResetSignal",
                             "native asynchronous reset must be a one-bit module signal",
                             async_reset->condition_location);
            return false;
        }
        const auto reset_net = reset_signal->bits.front();
        reset_active_low = async_reset->active_high
                               ? session.emit_not(reset_net)
                               : reset_net;
    }

    ResetLiteralValues async_reset_values;
    if (async_reset &&
        !find_reset_values(process.procedural_block->body.get(), *async_reset,
                           session, async_reset_values)) {
        session.diagnose("UnsupportedAsynchronousResetValue",
                         "reset branch must contain static constant assignments",
                         async_reset->condition_location);
        return false;
    }
    for (const auto* symbol : written_signals) {
        const auto* signal = session.signal(symbol);
        const auto& data_bits = next_state.at(symbol);
        const auto value_it = async_reset_values.bits.find(symbol);
        for (size_t bit = 0; bit < signal->bits.size(); ++bit) {
            const auto& q_net = signal->bits[bit];
            std::string cell_type = "DFF";
            std::vector<PinConnection> pins{{"D", data_bits[bit]},
                                            {"CLK", clock_net}};
            if (async_reset) {
                if (value_it == async_reset_values.bits.end() ||
                    bit >= value_it->second.size() || !value_it->second[bit]) {
                    session.diagnose("IncompleteAsynchronousResetValue",
                                     "register bit has no proven constant reset value",
                                     async_reset->condition_location);
                    return false;
                }
                if (*value_it->second[bit]) {
                    cell_type = "DFFS";
                    pins.push_back({"SET_B", reset_active_low});
                } else {
                    cell_type = "DFFR";
                    pins.push_back({"RESET_B", reset_active_low});
                }
            }
            const auto q_source = session.emit_cell(std::move(cell_type),
                                                    std::move(pins), "state");
            session.emit_buffer(q_source, q_net);
        }
    }
    return true;
}

std::optional<bool> lower_verilog_native_memory_process(
    VerilogNativeLoweringSession& session,
    const VerilogProcessAnalysis& analysis) {
    const auto& process = analysis.process;
    if (!process.procedural_block || !process.procedural_block->body)
        return std::nullopt;

    const auto* body = process.procedural_block->body.get();
    if (body->kind != VerilogStatementKind::Block || body->statements.size() != 2)
        return std::nullopt;

    const VerilogStatement* write = nullptr;
    const VerilogExpression* write_enable = nullptr;
    const auto* first = body->statements[0].get();
    if (first && first->kind == VerilogStatementKind::If && first->expression &&
        first->first_statement && !first->second_statement) {
        write_enable = first->expression.get();
        write = first->first_statement.get();
    } else {
        write = first;
    }
    const auto* read = body->statements[1].get();
    if (!write || !read ||
        write->kind != VerilogStatementKind::ProceduralAssignment ||
        read->kind != VerilogStatementKind::ProceduralAssignment ||
        write->assignment_kind != VerilogAssignmentKind::Nonblocking ||
        read->assignment_kind != VerilogAssignmentKind::Nonblocking ||
        !write->left || !write->right || !read->left || !read->right)
        return std::nullopt;

    auto memory_select = [&](const VerilogExpression& expression)
        -> std::pair<const VerilogNativeMemoryDeclaration*, const VerilogExpression*> {
        if (expression.kind != VerilogExpressionKind::Select ||
            expression.select_kind != VerilogSelectKind::Bit ||
            !expression.first || !expression.second ||
            expression.first->kind != VerilogExpressionKind::Identifier)
            return {nullptr, nullptr};
        const auto* symbol = session.resolve_symbol(*expression.first);
        const auto found = session.memory_declarations.find(symbol);
        return found == session.memory_declarations.end()
            ? std::pair<const VerilogNativeMemoryDeclaration*, const VerilogExpression*>{nullptr, nullptr}
            : std::pair<const VerilogNativeMemoryDeclaration*, const VerilogExpression*>{
                  &found->second, expression.second.get()};
    };

    const auto [write_memory, write_address_expression] = memory_select(*write->left);
    const auto [read_memory, read_address_expression] = memory_select(*read->right);
    if (!write_memory && !read_memory)
        return std::nullopt;
    if (!write_memory || !read_memory || write_memory->symbol != read_memory->symbol) {
        session.diagnose("UnsupportedNativeMemoryPorts",
                         "memory process must contain one whole-word write and one registered read of the same memory",
                         process.location);
        return false;
    }
    if (session.inferred_memories.contains(write_memory->symbol)) {
        session.diagnose("MultipleNativeMemoryWriters",
                         "native memory inference supports exactly one clocked process per memory",
                         process.location);
        return false;
    }
    if (process.events.size() != 1 ||
        process.events.front().edge != VerilogEventEdge::Posedge ||
        !process.events.front().symbol) {
        session.diagnose("UnsupportedNativeMemoryClock",
                         "native memory inference requires one positive-edge clock",
                         process.location);
        return false;
    }
    const auto* clock_signal = session.signal(process.events.front().symbol);
    if (!clock_signal || clock_signal->bits.size() != 1) {
        session.diagnose("UnsupportedNativeMemoryClock",
                         "memory clock must resolve to a one-bit module input",
                         process.events.front().location);
        return false;
    }

    const size_t word_width = *write_memory->word_type.bit_width;
    const size_t depth = write_memory->depth;
    if ((depth & (depth - 1)) != 0) {
        session.diagnose("UnsupportedNativeMemoryDepth",
                         "native memory inference currently requires a power-of-two depth",
                         write_memory->declaration->location);
        return false;
    }
    size_t address_width = 0;
    for (size_t values = depth; values > 1; values >>= 1)
        ++address_width;

    auto write_address = lower_verilog_native_expression(
        session, *write_address_expression);
    auto read_address = lower_verilog_native_expression(
        session, *read_address_expression);
    auto write_data = lower_verilog_native_expression(
        session, *write->right, {}, word_width);
    if (!write_address || !read_address || !write_data ||
        write_address->bits.size() != address_width ||
        read_address->bits.size() != address_width ||
        write_data->bits.size() != word_width) {
        session.diagnose("UnsupportedNativeMemoryPortWidth",
                         "memory address or data width does not match the inferred declaration",
                         process.location);
        return false;
    }

    const auto* output_symbol = read->left->kind == VerilogExpressionKind::Identifier
        ? session.resolve_symbol(*read->left) : nullptr;
    const auto* output_signal = session.signal(output_symbol);
    if (!output_signal || output_signal->bits.size() != word_width) {
        session.diagnose("UnsupportedNativeMemoryReadTarget",
                         "registered memory read must assign one complete output word",
                         read->left->location);
        return false;
    }
    std::string enable_net = session.constant(true);
    if (write_enable) {
        auto enable = lower_verilog_native_expression(session, *write_enable);
        if (!enable || enable->bits.size() != 1) {
            session.diagnose("UnsupportedNativeMemoryWriteEnable",
                             "memory write enable must be a one-bit expression",
                             write_enable->location);
            return false;
        }
        enable_net = enable->bits.front();
    }

    session.memories.emplace_back(
        write_memory->symbol->name, word_width, depth, address_width,
        clock_signal->bits.front(), enable_net, std::move(write_address->bits),
        std::move(write_data->bits), std::move(read_address->bits),
        output_signal->bits, MemoryReadDuringWrite::ReadBeforeWrite);
    session.inferred_memories.insert(write_memory->symbol);
    session.driven_net_names.insert(output_signal->bits.begin(),
                                    output_signal->bits.end());
    return true;
}

} // namespace labneura::eda::detail
