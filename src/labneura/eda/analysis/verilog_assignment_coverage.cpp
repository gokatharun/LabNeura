#include "labneura/eda/verilog_process_analysis.h"

#include "labneura/eda/verilog_assignment.h"
#include "labneura/eda/verilog_constant.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <limits>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace labneura::eda {
namespace {

using Symbol = VerilogSemanticSymbol;
using Coverage = VerilogAssignmentCoverageState;

enum class Flow : uint8_t { Normal, Break, Continue, Return, Unknown };

struct PathState {
    Flow flow = Flow::Normal;
    std::unordered_map<const Symbol*, std::vector<uint8_t>> assigned;
    std::unordered_set<const Symbol*> unknown;
};

bool is_process_signal(const VerilogProcess& process,
                       const VerilogSemanticSymbol* symbol) {
    if (!symbol || (symbol->kind != VerilogSemanticSymbolKind::Port &&
                    symbol->kind != VerilogSemanticSymbolKind::Net &&
                    symbol->kind != VerilogSemanticSymbolKind::Variable))
        return false;
    if (!process.module_symbol || !process.module_symbol->member_scope)
        return true;
    for (const auto* scope = symbol->scope; scope; scope = scope->parent)
        if (scope == process.module_symbol->member_scope)
            return true;
    return false;
}

const Symbol* symbol_at(const VerilogExpression& expression,
                        const VerilogSemanticAnalysisResult& semantic) {
    for (const auto& entry : semantic.references_at(expression.location)) {
        const auto& reference = semantic.references[entry.reference_ordinal];
        if (reference.declaration)
            return reference.declaration;
    }

    const Symbol* result = nullptr;
    for (const auto& reference : semantic.references) {
        if (!reference.declaration || reference.name != expression.text)
            continue;
        if (result && result != reference.declaration)
            return nullptr;
        result = reference.declaration;
    }
    return result;
}

const Symbol* lvalue_root_symbol(
    const VerilogExpression& expression,
    const VerilogSemanticAnalysisResult& semantic) {
    switch (expression.kind) {
        case VerilogExpressionKind::Identifier:
            return symbol_at(expression, semantic);
        case VerilogExpressionKind::Select:
        case VerilogExpressionKind::MemberAccess:
            return expression.first ? lvalue_root_symbol(*expression.first, semantic)
                                    : nullptr;
        default:
            return nullptr;
    }
}

class CoverageAnalyzer {
public:
    CoverageAnalyzer(const VerilogProcess& process,
                     const VerilogSemanticAnalysisResult& semantic,
                     const VerilogTypeAnalysisResult& types)
        : process_(process), semantic_(semantic), types_(types) {}

    VerilogProcessAssignmentCoverage run() {
        std::vector<PathState> paths(1);
        if (process_.continuous_assignment) {
            apply_assignment(process_.continuous_assignment->left.get(),
                             paths.front(), process_.continuous_assignment->location);
        } else if (process_.procedural_block &&
                   process_.procedural_block->body) {
            paths = execute(*process_.procedural_block->body, std::move(paths), 0);
        } else {
            result_.knowledge = VerilogAnalysisKnowledge::Unknown;
            add_diagnostic("MissingProcessBody",
                           "assignment coverage requires a process body",
                           process_.location);
        }

        // Control transfers that escape the process cannot be credited as a
        // successful complete assignment path.
        for (auto& path : paths) {
            if (path.flow == Flow::Break || path.flow == Flow::Continue ||
                path.flow == Flow::Unknown) {
                path.unknown.insert(all_targets_.begin(), all_targets_.end());
                result_.knowledge = VerilogAnalysisKnowledge::Unknown;
            }
            path.flow = Flow::Normal;
        }
        summarize(paths);
        return std::move(result_);
    }

private:
    static constexpr size_t kMaximumPaths = 96;

    const VerilogProcess& process_;
    const VerilogSemanticAnalysisResult& semantic_;
    const VerilogTypeAnalysisResult& types_;
    VerilogProcessAssignmentCoverage result_;
    std::vector<const Symbol*> all_targets_;
    std::unordered_map<const Symbol*, size_t> widths_;
    std::unordered_set<const Symbol*> target_set_;

    void add_diagnostic(std::string code, std::string message,
                        const VerilogSyntaxLocation& location) {
        result_.diagnostics.push_back(
            {std::move(code), std::move(message), location, std::nullopt});
    }

    std::optional<VerilogConstantValue> evaluate_constant(
        const VerilogExpression& expression) const {
        const auto resolve = [&](std::string_view name,
                                 const VerilogSyntaxLocation& location)
            -> std::optional<VerilogConstantValue> {
            for (const auto& entry : semantic_.references_at(location)) {
                const auto& reference =
                    semantic_.references[entry.reference_ordinal];
                if (reference.name != name || !reference.declaration)
                    continue;
                const auto value = types_.symbol_constants.find(reference.declaration);
                if (value != types_.symbol_constants.end())
                    return value->second;
            }
            const Symbol* fallback = nullptr;
            for (const auto& item : semantic_.references) {
                if (item.name != name || !item.declaration)
                    continue;
                const auto value = types_.symbol_constants.find(item.declaration);
                if (value == types_.symbol_constants.end())
                    continue;
                if (fallback && fallback != item.declaration)
                    return std::nullopt;
                fallback = item.declaration;
            }
            if (fallback) {
                const auto value = types_.symbol_constants.find(fallback);
                if (value != types_.symbol_constants.end())
                    return value->second;
            }
            return std::nullopt;
        };
        const auto evaluated = evaluate_verilog_constant(expression, {}, resolve);
        return evaluated.value;
    }

    std::optional<bool> constant_truth(const VerilogExpression* expression) const {
        if (!expression)
            return std::nullopt;
        const auto value = evaluate_constant(*expression);
        if (!value || value->bits.empty() || !value->is_fully_known())
            return std::nullopt;
        return std::any_of(value->bits.begin(), value->bits.end(),
                           [](VerilogLogicBit bit) {
                               return bit == VerilogLogicBit::One;
                           });
    }

    static bool case_match(const VerilogConstantValue& selector,
                           const VerilogConstantValue& item,
                           VerilogCaseKind kind) {
        const auto width = std::max(selector.bits.size(), item.bits.size());
        const bool sign_extend = selector.is_signed && item.is_signed;
        for (size_t bit = 0; bit < width; ++bit) {
            const auto lhs = bit < selector.bits.size()
                                 ? selector.bits[bit]
                                 : sign_extend && !selector.bits.empty()
                                       ? selector.bits.back()
                                       : VerilogLogicBit::Zero;
            const auto rhs = bit < item.bits.size()
                                 ? item.bits[bit]
                                 : sign_extend && !item.bits.empty()
                                       ? item.bits.back()
                                       : VerilogLogicBit::Zero;
            const bool lhs_wildcard =
                (kind == VerilogCaseKind::CaseZ && lhs == VerilogLogicBit::Z) ||
                (kind == VerilogCaseKind::CaseX &&
                 (lhs == VerilogLogicBit::X || lhs == VerilogLogicBit::Z));
            const bool rhs_wildcard =
                (kind == VerilogCaseKind::CaseZ && rhs == VerilogLogicBit::Z) ||
                (kind == VerilogCaseKind::CaseX &&
                 (rhs == VerilogLogicBit::X || rhs == VerilogLogicBit::Z));
            if (!lhs_wildcard && !rhs_wildcard && lhs != rhs)
                return false;
        }
        return true;
    }

    std::optional<const VerilogCaseItem*> select_constant_case(
        const VerilogStatement& statement) const {
        if (!statement.expression)
            return std::nullopt;
        const auto selector = evaluate_constant(*statement.expression);
        if (!selector)
            return std::nullopt;
        const VerilogCaseItem* default_item = nullptr;
        for (const auto& item : statement.case_items) {
            if (item.is_default) {
                if (!default_item)
                    default_item = &item;
                continue;
            }
            for (const auto& expression : item.expressions) {
                if (!expression)
                    continue;
                const auto candidate = evaluate_constant(*expression);
                if (candidate && case_match(*selector, *candidate,
                                            statement.case_kind))
                    return &item;
            }
        }
        return default_item;
    }

    void register_target(const Symbol* symbol) {
        if (!is_process_signal(process_, symbol))
            return;
        if (target_set_.insert(symbol).second) {
            all_targets_.push_back(symbol);
            const auto type = types_.symbol_types.find(symbol);
            widths_[symbol] =
                type != types_.symbol_types.end() && type->second.bit_width
                    ? *type->second.bit_width
                    : 0;
        }
    }

    void mark_unknown(PathState& path, const Symbol* symbol) {
        if (!is_process_signal(process_, symbol))
            return;
        register_target(symbol);
        path.unknown.insert(symbol);
    }

    void apply_assignment(const VerilogExpression* lvalue, PathState& path,
                          const VerilogSyntaxLocation& location) {
        if (!lvalue) {
            result_.knowledge = VerilogAnalysisKnowledge::Unknown;
            add_diagnostic("MissingAssignmentLvalue",
                           "assignment coverage found an assignment without an lvalue",
                           location);
            return;
        }
        const auto resolved = resolve_verilog_lvalue(
            *lvalue, semantic_, types_,
            [&](const VerilogExpression& expression) {
                return evaluate_constant(expression);
            });
        if (!resolved.target) {
            const auto* root = lvalue_root_symbol(*lvalue, semantic_);
            mark_unknown(path, root);
            result_.knowledge = VerilogAnalysisKnowledge::Unknown;
            add_diagnostic("UnknownAssignmentCoverage",
                           "assignment destination could not be resolved to constant bit coverage",
                           lvalue->location);
            return;
        }

        for (const auto& destination : resolved.target->destinations) {
            const auto* symbol = destination.symbol;
            if (!is_process_signal(process_, symbol))
                continue;
            register_target(symbol);
            const auto width = widths_[symbol];
            if (width == 0) {
                mark_unknown(path, symbol);
                result_.knowledge = VerilogAnalysisKnowledge::Unknown;
                continue;
            }
            auto& bits = path.assigned[symbol];
            if (bits.empty())
                bits.resize(width, 0);
            for (const auto position : destination.bit_positions) {
                if (!position)
                    continue;
                if (*position >= bits.size()) {
                    mark_unknown(path, symbol);
                    result_.knowledge = VerilogAnalysisKnowledge::Unknown;
                } else {
                    bits[*position] = 1;
                }
            }
        }
    }

    std::vector<PathState> execute(const VerilogStatement& statement,
                                  std::vector<PathState> paths,
                                  size_t loop_depth) {
        if (paths.empty())
            return paths;
        switch (statement.kind) {
            case VerilogStatementKind::Block:
                for (const auto& child : statement.statements) {
                    if (!child)
                        continue;
                    std::vector<PathState> active;
                    std::vector<PathState> terminal;
                    for (auto& path : paths) {
                        if (path.flow == Flow::Normal)
                            active.push_back(std::move(path));
                        else
                            terminal.push_back(std::move(path));
                    }
                    if (active.empty())
                        return terminal;
                    auto next = execute(*child, std::move(active), loop_depth);
                    terminal.insert(terminal.end(),
                                    std::make_move_iterator(next.begin()),
                                    std::make_move_iterator(next.end()));
                    paths = std::move(terminal);
                    paths = limit_paths(std::move(paths), statement.location);
                }
                return paths;
            case VerilogStatementKind::ProceduralAssignment:
                for (auto& path : paths)
                    if (path.flow == Flow::Normal)
                        apply_assignment(statement.left.get(), path,
                                         statement.location);
                return paths;
            case VerilogStatementKind::If:
                return execute_if(statement, std::move(paths), loop_depth);
            case VerilogStatementKind::Case:
                return execute_case(statement, std::move(paths), loop_depth);
            case VerilogStatementKind::Loop:
                return execute_loop(statement, std::move(paths), loop_depth);
            case VerilogStatementKind::Break:
                for (auto& path : paths)
                    if (path.flow == Flow::Normal)
                        path.flow = loop_depth == 0 ? Flow::Unknown : Flow::Break;
                return paths;
            case VerilogStatementKind::Continue:
                for (auto& path : paths)
                    if (path.flow == Flow::Normal)
                        path.flow = loop_depth == 0 ? Flow::Unknown : Flow::Continue;
                return paths;
            case VerilogStatementKind::Return:
                for (auto& path : paths)
                    if (path.flow == Flow::Normal)
                        path.flow = Flow::Return;
                return paths;
            case VerilogStatementKind::EventControl:
            case VerilogStatementKind::DelayControl:
            case VerilogStatementKind::Wait:
            case VerilogStatementKind::TaskCall:
            case VerilogStatementKind::Unsupported:
                result_.knowledge = VerilogAnalysisKnowledge::Unsupported;
                add_diagnostic("UnsupportedCoverageStatement",
                               "assignment coverage is incomplete for this procedural statement",
                               statement.location);
                for (auto& path : paths)
                    if (path.flow == Flow::Normal)
                        path.flow = Flow::Unknown;
                return paths;
            case VerilogStatementKind::Empty:
                return paths;
        }
        result_.knowledge = VerilogAnalysisKnowledge::Unknown;
        for (auto& path : paths)
            if (path.flow == Flow::Normal)
                path.flow = Flow::Unknown;
        return paths;
    }

    std::vector<PathState> execute_if(const VerilogStatement& statement,
                                     std::vector<PathState> paths,
                                     size_t loop_depth) {
        const auto condition = constant_truth(statement.expression.get());
        std::vector<PathState> then_paths;
        std::vector<PathState> else_paths;
        for (auto& path : paths) {
            if (path.flow != Flow::Normal) {
                else_paths.push_back(std::move(path));
            } else if (condition && *condition) {
                then_paths.push_back(std::move(path));
            } else if (condition && !*condition) {
                else_paths.push_back(std::move(path));
            } else {
                then_paths.push_back(path);
                else_paths.push_back(std::move(path));
            }
        }
        if (statement.first_statement && !then_paths.empty())
            then_paths = execute(*statement.first_statement,
                                 std::move(then_paths), loop_depth);
        if (statement.second_statement && !else_paths.empty())
            else_paths = execute(*statement.second_statement,
                                 std::move(else_paths), loop_depth);
        then_paths.insert(then_paths.end(),
                          std::make_move_iterator(else_paths.begin()),
                          std::make_move_iterator(else_paths.end()));
        return limit_paths(std::move(then_paths), statement.location);
    }

    std::vector<PathState> execute_case(const VerilogStatement& statement,
                                       std::vector<PathState> paths,
                                       size_t loop_depth) {
        const auto selected = select_constant_case(statement);
        if (selected) {
            std::vector<PathState> result;
            const VerilogStatement* body =
                *selected ? (*selected)->statement.get() : nullptr;
            for (auto& path : paths) {
                if (path.flow != Flow::Normal || !body)
                    result.push_back(std::move(path));
                else {
                    auto branch = execute(*body, {std::move(path)}, loop_depth);
                    result.insert(result.end(),
                                  std::make_move_iterator(branch.begin()),
                                  std::make_move_iterator(branch.end()));
                }
            }
            return result;
        }

        std::vector<PathState> alternatives;
        const bool has_default = std::any_of(
            statement.case_items.begin(), statement.case_items.end(),
            [](const auto& item) { return item.is_default; });
        for (const auto& item : statement.case_items) {
            std::vector<PathState> branch_inputs;
            for (const auto& path : paths)
                if (path.flow == Flow::Normal)
                    branch_inputs.push_back(path);
            auto branch = item.statement
                              ? execute(*item.statement, std::move(branch_inputs),
                                        loop_depth)
                              : std::move(branch_inputs);
            alternatives.insert(alternatives.end(),
                                std::make_move_iterator(branch.begin()),
                                std::make_move_iterator(branch.end()));
        }
        for (auto& path : paths)
            if (path.flow != Flow::Normal || !has_default)
                alternatives.push_back(std::move(path));
        return limit_paths(std::move(alternatives), statement.location);
    }

    std::vector<PathState> execute_loop(const VerilogStatement& statement,
                                       std::vector<PathState> paths,
                                       size_t loop_depth) {
        if (statement.loop_kind == VerilogLoopKind::Repeat) {
            const auto count_value = statement.expression
                                         ? evaluate_constant(*statement.expression)
                                         : std::nullopt;
            const auto count = count_value && count_value->is_fully_known()
                                   ? count_value->to_uint64()
                                   : std::nullopt;
            if (count && *count == 0)
                return paths;
            if (count && *count > 0 && statement.first_statement) {
                auto once = execute(*statement.first_statement, std::move(paths),
                                    loop_depth + 1);
                for (auto& path : once)
                    if (path.flow == Flow::Break ||
                        path.flow == Flow::Continue)
                        path.flow = Flow::Normal;
                if (*count > 1) {
                    auto twice = execute(*statement.first_statement,
                                         std::move(once), loop_depth + 1);
                    for (auto& path : twice)
                        if (path.flow == Flow::Break ||
                            path.flow == Flow::Continue)
                            path.flow = Flow::Normal;
                    if (*count > 2) {
                        result_.knowledge = VerilogAnalysisKnowledge::Unknown;
                        for (auto& path : twice)
                            for (const auto* target : all_targets_)
                                path.unknown.insert(target);
                    }
                    return limit_paths(std::move(twice), statement.location);
                }
                return once;
            }
        } else if (statement.loop_kind == VerilogLoopKind::DoWhile &&
                   statement.first_statement) {
            auto once = execute(*statement.first_statement, std::move(paths),
                                loop_depth + 1);
            for (auto& path : once)
                if (path.flow == Flow::Break ||
                    path.flow == Flow::Continue)
                    path.flow = Flow::Normal;
            return once;
        }

        if (statement.loop_kind == VerilogLoopKind::Forever) {
            result_.knowledge = VerilogAnalysisKnowledge::Unknown;
            add_diagnostic("UnboundedLoopCoverageUnknown",
                           "forever-loop assignment coverage is unknown without a statically bounded exit",
                           statement.location);
            for (auto& path : paths)
                for (const auto* target : all_targets_)
                    path.unknown.insert(target);
            return paths;
        }

        // A for-loop initializer executes even when the loop body executes
        // zero times. The body itself remains optional unless statically
        // proven otherwise.
        if (statement.loop_kind == VerilogLoopKind::For &&
            statement.loop_initialization)
            paths = execute(*statement.loop_initialization, std::move(paths),
                            loop_depth);

        // While and for loops may execute zero times. Add body paths for
        // possible writes but keep the zero-iteration path for completeness.
        std::vector<PathState> alternatives = paths;
        if (statement.first_statement) {
            auto body = execute(*statement.first_statement, paths, loop_depth + 1);
            for (auto& path : body)
                if (path.flow == Flow::Break ||
                    path.flow == Flow::Continue)
                    path.flow = Flow::Normal;
            alternatives.insert(alternatives.end(),
                                std::make_move_iterator(body.begin()),
                                std::make_move_iterator(body.end()));
        }
        return limit_paths(std::move(alternatives), statement.location);
    }

    std::vector<PathState> limit_paths(std::vector<PathState> paths,
                                       const VerilogSyntaxLocation& location) {
        if (paths.size() <= kMaximumPaths)
            return paths;
        result_.knowledge = VerilogAnalysisKnowledge::Unknown;
        add_diagnostic("CoveragePathLimitExceeded",
                       "assignment coverage exceeded its conservative path limit",
                       location);
        PathState merged;
        merged.flow = Flow::Unknown;
        for (const auto* symbol : all_targets_) {
            const size_t width = widths_[symbol];
            std::vector<uint8_t> intersection(width, 1);
            for (const auto& path : paths) {
                const auto found = path.assigned.find(symbol);
                for (size_t bit = 0; bit < width; ++bit)
                    if (found == path.assigned.end() ||
                        bit >= found->second.size() || !found->second[bit])
                        intersection[bit] = 0;
                if (path.unknown.contains(symbol))
                    merged.unknown.insert(symbol);
            }
            merged.assigned[symbol] = std::move(intersection);
            merged.unknown.insert(symbol);
        }
        return {std::move(merged)};
    }

    void summarize(const std::vector<PathState>& paths) {
        for (const auto* symbol : all_targets_) {
            VerilogSymbolAssignmentCoverage summary;
            summary.symbol = symbol;
            summary.bit_width = widths_[symbol];
            const size_t width = summary.bit_width;
            summary.bits.resize(width, Coverage::Unknown);
            if (width == 0 || paths.empty()) {
                summary.whole_signal = Coverage::Unknown;
                result_.knowledge = VerilogAnalysisKnowledge::Unknown;
                result_.symbols.push_back(std::move(summary));
                continue;
            }

            bool any_assigned = false;
            bool every_bit_definite = true;
            bool symbol_unknown = false;
            for (size_t bit = 0; bit < width; ++bit) {
                bool all_paths_assigned = true;
                bool some_path_assigned = false;
                bool bit_unknown = false;
                for (const auto& path : paths) {
                    if (path.unknown.contains(symbol)) {
                        bit_unknown = true;
                        symbol_unknown = true;
                        all_paths_assigned = false;
                        continue;
                    }
                    const auto found = path.assigned.find(symbol);
                    const bool assigned = found != path.assigned.end() &&
                                          bit < found->second.size() &&
                                          found->second[bit] != 0;
                    all_paths_assigned &= assigned;
                    some_path_assigned |= assigned;
                }
                any_assigned |= some_path_assigned;
                if (bit_unknown) {
                    summary.bits[bit] = Coverage::Unknown;
                    every_bit_definite = false;
                } else if (all_paths_assigned) {
                    summary.bits[bit] = Coverage::DefinitelyAssigned;
                } else if (some_path_assigned) {
                    summary.bits[bit] = Coverage::PossiblyAssigned;
                    every_bit_definite = false;
                } else {
                    summary.bits[bit] = Coverage::NotAssigned;
                    every_bit_definite = false;
                }
            }

            if (symbol_unknown) {
                summary.whole_signal = Coverage::Unknown;
            } else if (every_bit_definite) {
                summary.whole_signal = Coverage::DefinitelyAssigned;
            } else if (any_assigned) {
                summary.whole_signal = Coverage::PossiblyAssigned;
            } else {
                summary.whole_signal = Coverage::NotAssigned;
            }
            if (summary.whole_signal != Coverage::DefinitelyAssigned &&
                summary.whole_signal != Coverage::Unknown &&
                any_assigned &&
                (process_.kind == VerilogProcessKind::AlwaysComb ||
                 process_.kind == VerilogProcessKind::AlwaysLatch ||
                 (process_.kind == VerilogProcessKind::Always &&
                  process_.wildcard_sensitivity)))
                result_.may_require_storage = true;
            result_.symbols.push_back(std::move(summary));
        }
        std::sort(result_.symbols.begin(), result_.symbols.end(),
                  [](const auto& lhs, const auto& rhs) {
                      if (lhs.symbol->declaration_location.byte_offset !=
                          rhs.symbol->declaration_location.byte_offset)
                          return lhs.symbol->declaration_location.byte_offset <
                                 rhs.symbol->declaration_location.byte_offset;
                      return lhs.symbol->name < rhs.symbol->name;
                  });
    }
};

} // namespace

VerilogProcessAssignmentCoverage analyze_verilog_assignment_coverage(
    const VerilogProcess& process,
    const VerilogSemanticAnalysisResult& semantic,
    const VerilogTypeAnalysisResult& type_analysis) {
    return CoverageAnalyzer(process, semantic, type_analysis).run();
}

} // namespace labneura::eda
