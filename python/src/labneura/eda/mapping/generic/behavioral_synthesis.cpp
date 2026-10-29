// Behavioral RTL synthesis: lowers an elaborated slang AST (always blocks,
// continuous assigns, generate blocks, memories, hierarchy) into a bit-level
// generic netlist (AND2/OR2/XOR2/INV/MUX2/DFF/TIE) that technology mapping
// can consume.
#include "labneura/eda/netlist.h"

#include "behavioral_synthesis.h"
#include "labneura/eda/parsed_verilog_context.h"
#include "bit_graph.h"

#include "slang/ast/Compilation.h"
#include "slang/ast/Symbol.h"
#include "slang/ast/symbols/AttributeSymbol.h"
#include "slang/ast/EvalContext.h"
#include "slang/ast/Statement.h"
#include "slang/ast/TimingControl.h"
#include "slang/ast/expressions/AssignmentExpressions.h"
#include "slang/ast/expressions/CallExpression.h"
#include "slang/ast/expressions/ConversionExpression.h"
#include "slang/ast/expressions/LiteralExpressions.h"
#include "slang/ast/expressions/MiscExpressions.h"
#include "slang/ast/expressions/OperatorExpressions.h"
#include "slang/ast/expressions/SelectExpressions.h"
#include "slang/ast/statements/ConditionalStatements.h"
#include "slang/ast/statements/LoopStatements.h"
#include "slang/ast/statements/MiscStatements.h"
#include "slang/ast/symbols/BlockSymbols.h"
#include "slang/ast/symbols/CompilationUnitSymbols.h"
#include "slang/ast/symbols/InstanceSymbols.h"
#include "slang/ast/symbols/MemberSymbols.h"
#include "slang/ast/symbols/PortSymbols.h"
#include "slang/ast/symbols/SubroutineSymbols.h"
#include "slang/ast/symbols/VariableSymbols.h"
#include "slang/ast/types/AllTypes.h"
#include "slang/ast/types/Type.h"
#include "slang/diagnostics/TextDiagnosticClient.h"
#include "slang/driver/Driver.h"

#include <algorithm>
#include <span>
#include <functional>
#include <map>
#include <mutex>
#include <memory>
#include <mutex>
#include <set>
#include <unordered_map>
#include <unordered_set>

namespace labneura::eda {
namespace {

namespace sa = slang::ast;
using synth::Bit;
using synth::Node;
using synth::BitGraph;
using synth::kLatch;
using synth::kOne;
using synth::kZero;
using synth::Op;
using synth::Word;
using synth::WordOps;

[[noreturn]] void unsupported(const std::string& what) {
    throw std::runtime_error("behavioral synthesis: unsupported " + what);
}

struct Var {
    const sa::ValueSymbol* symbol = nullptr;
    std::string name;
    std::size_t width = 0;
    std::size_t element_width = 0;  // arrays only
    std::size_t element_count = 0;  // arrays only
    bool is_array = false;
    bool is_scalar = false;
    bool registered = false;
    slang::ConstantRange range;  // packed range, or array index range
    Word wires;
    std::vector<char> driven;
};

struct Target {
    Var* var;
    std::size_t bit;
    Bit condition;
};

struct State {
    std::unordered_map<const sa::Symbol*, Word> cur;
    std::unordered_map<const sa::Symbol*, Word> nb;
    Bit returned = 0;  // condition under which a function `return` already executed
};

struct Candidate {
    Bit condition;
    long offset;
};

long signed_value(const Word& word, bool is_signed) {
    const auto value = synth::as_constant(word);
    if (!value) unsupported("non-constant value where a constant is required");
    std::int64_t result = static_cast<std::int64_t>(*value);
    if (is_signed && !word.empty() && word.size() < 64 && word.back() == kOne) {
        result -= std::int64_t{1} << word.size();
    }
    return static_cast<long>(result);
}

std::size_t type_width(const sa::Type& type) {
    // Unpacked arrays report no bit width; they are stored flat, element by element.
    const auto& canonical = type.getCanonicalType();
    if (canonical.isUnpackedArray() && canonical.hasFixedRange()) {
        const auto* element = canonical.getArrayElementType();
        return element ? canonical.getFixedRange().width() * type_width(*element) : 0;
    }
    return type.getBitWidth();
}

class Synthesizer {
public:
    Synthesizer() : ops_(graph_) {}

    Module run(const sa::InstanceSymbol& top, const std::string& top_name);

private:
    BitGraph graph_;
    WordOps ops_;
    std::unordered_map<const sa::Symbol*, Var> vars_;
    std::string top_prefix_;
    const sa::Symbol* anchor_ = nullptr;
    State st_;
    bool seq_ = false;
    int call_depth_ = 0;
    Var* return_var_ = nullptr;
    const sa::Expression* lvalue_reference_ = nullptr;

    struct Output {
        std::string name;
        Var* var;
        std::size_t bit;
    };
    std::vector<const sa::ProceduralBlockSymbol*> procedures_;
    std::vector<const sa::ContinuousAssignSymbol*> assigns_;
    std::vector<const sa::NetSymbol*> net_initializers_;
    std::vector<const sa::InstanceSymbol*> instances_;

    // ---- variables ----
    std::string path_name(const sa::Symbol& symbol) const {
        std::string path = symbol.getHierarchicalPath();
        if (path.rfind(top_prefix_, 0) == 0) path.erase(0, top_prefix_.size());
        return path;
    }

    Var& register_var(const sa::ValueSymbol& symbol, bool module_level) {
        auto [it, inserted] = vars_.try_emplace(&symbol);
        Var& var = it->second;
        if (!inserted) return var;
        const auto& type = symbol.getType();
        var.symbol = &symbol;
        var.name = module_level ? path_name(symbol) : std::string(symbol.name);
        var.registered = module_level;
        if (type.isUnpackedArray()) {
            const auto* element = type.getArrayElementType();
            if (!element || !element->isIntegral() || !type.hasFixedRange()) {
                unsupported("array type of '" + var.name + "'");
            }
            var.is_array = true;
            var.range = type.getFixedRange();
            var.element_width = type_width(*element);
            var.element_count = var.range.width();
            var.width = var.element_width * var.element_count;
        } else if (type.isIntegral()) {
            var.width = type_width(type);
            var.is_scalar = type.isScalar();
            var.range = type.hasFixedRange()
                            ? type.getFixedRange()
                            : slang::ConstantRange{static_cast<int32_t>(var.width) - 1, 0};
        } else {
            unsupported("type of '" + var.name + "'");
        }
        var.driven.assign(var.width, 0);
        if (module_level) {
            for (std::size_t i = 0; i < var.width; ++i) {
                const Bit wire = graph_.wire();
                graph_.node(wire).name = bit_name(var, i);
                var.wires.push_back(wire);
            }
        } else {
            var.wires.assign(var.width, kLatch);
        }
        return var;
    }

    Var& var_for(const sa::ValueSymbol& symbol) {
        const auto it = vars_.find(&symbol);
        if (it != vars_.end()) return it->second;
        return register_var(symbol, false);
    }

    std::string bit_name(const Var& var, std::size_t bit) const {
        if (var.is_scalar && var.width == 1) return var.name;
        if (var.is_array) {
            return var.name + "[" + std::to_string(bit / var.element_width +
                                                   var.range.lower()) +
                   "][" + std::to_string(bit % var.element_width) + "]";
        }
        const long logical = var.range.left >= var.range.right
                                 ? static_cast<long>(var.range.right) + static_cast<long>(bit)
                                 : static_cast<long>(var.range.right) - static_cast<long>(bit);
        return var.name + "[" + std::to_string(logical) + "]";
    }

    void drive(Var& var, std::size_t bit, Bit value) {
        if (var.driven[bit]) {
            throw std::runtime_error("behavioral synthesis: multiple drivers on " +
                                     bit_name(var, bit));
        }
        var.driven[bit] = 1;
        graph_.node(var.wires[bit]).a = value;
    }

    // ---- scope walking ----
    void collect(const sa::Scope& scope) {
        for (const auto& member : scope.members()) {
            switch (member.kind) {
                case sa::SymbolKind::Net: {
                    const auto& net = member.as<sa::NetSymbol>();
                    register_var(net, true);
                    if (net.getInitializer()) net_initializers_.push_back(&net);
                    break;
                }
                case sa::SymbolKind::Variable:
                    register_var(member.as<sa::VariableSymbol>(), true);
                    break;
                case sa::SymbolKind::ProceduralBlock:
                    procedures_.push_back(&member.as<sa::ProceduralBlockSymbol>());
                    break;
                case sa::SymbolKind::ContinuousAssign:
                    assigns_.push_back(&member.as<sa::ContinuousAssignSymbol>());
                    break;
                case sa::SymbolKind::GenerateBlock: {
                    const auto& block = member.as<sa::GenerateBlockSymbol>();
                    if (!block.isUninstantiated) collect(block);
                    break;
                }
                case sa::SymbolKind::GenerateBlockArray: {
                    const auto& array = member.as<sa::GenerateBlockArraySymbol>();
                    for (const auto* entry : array.entries) {
                        if (!entry->isUninstantiated) collect(*entry);
                    }
                    break;
                }
                case sa::SymbolKind::Instance: {
                    const auto& child = member.as<sa::InstanceSymbol>();
                    instances_.push_back(&child);
                    collect(child.body);
                    break;
                }
                case sa::SymbolKind::PrimitiveInstance:
                    unsupported("gate primitive instances in behavioral synthesis");
                default:
                    break;
            }
        }
    }

    // ---- constants ----
    Word from_constant(const slang::ConstantValue& value, std::size_t width,
                       bool is_signed) {
        if (!value.isInteger()) unsupported("non-integer constant");
        const auto& integer = value.integer();
        Word word(integer.getBitWidth());
        for (std::size_t i = 0; i < word.size(); ++i) {
            const auto bit = integer[static_cast<int32_t>(i)];
            word[i] = !bit.isUnknown() && bit.value == 1 ? kOne : kZero;
        }
        return synth::resize(word, width, is_signed && integer.isSigned());
    }

    std::optional<Word> constant_of(const sa::Expression& expr) {
        if (expr.type->isIntegral() == false) return std::nullopt;
        if (const auto* constant = expr.getConstant()) {
            if (constant->isInteger()) {
                return from_constant(*constant, type_width(*expr.type),
                                     expr.type->isSigned());
            }
        }
        const bool try_eval =
            (expr.kind == sa::ExpressionKind::Call &&
             expr.as<sa::CallExpression>().isSystemCall()) ||
            expr.kind == sa::ExpressionKind::IntegerLiteral ||
            expr.kind == sa::ExpressionKind::UnbasedUnsizedIntegerLiteral ||
            (expr.kind == sa::ExpressionKind::NamedValue &&
             (expr.as<sa::NamedValueExpression>().symbol.kind ==
                  sa::SymbolKind::Parameter ||
              expr.as<sa::NamedValueExpression>().symbol.kind ==
                  sa::SymbolKind::EnumValue));
        if (!try_eval) return std::nullopt;
        sa::EvalContext context(*anchor_);
        const auto value = expr.eval(context);
        if (!value.isInteger()) return std::nullopt;
        return from_constant(value, type_width(*expr.type), expr.type->isSigned());
    }

    long constant_int(const sa::Expression& expr) {
        const auto word = eval(expr);
        return signed_value(word, expr.type->isSigned());
    }

    // ---- reads ----
    const Word& read_var(Var& var) {
        const auto found = st_.cur.find(var.symbol);
        return found != st_.cur.end() ? found->second : var.wires;
    }

    static long translate(const slang::ConstantRange& range, long index) {
        return range.left >= range.right ? index - range.right
                                         : static_cast<long>(range.right) - index;
    }

    std::vector<Candidate> element_candidates(const sa::Expression& selector,
                                              const slang::ConstantRange& range,
                                              std::size_t unit_width, bool is_array) {
        const Word index = eval(selector);
        const auto position = [&](long logical) {
            return is_array ? logical - range.lower() : translate(range, logical);
        };
        std::vector<Candidate> result;
        if (synth::as_constant(index)) {
            const long logical = signed_value(index, selector.type->isSigned());
            result.push_back({kOne, position(logical) * static_cast<long>(unit_width)});
            return result;
        }
        for (long k = range.lower(); k <= range.upper(); ++k) {
            const Bit condition = ops_.equal(
                index, synth::constant_word(static_cast<std::uint64_t>(k), index.size()));
            if (condition == kZero) continue;
            result.push_back({condition, position(k) * static_cast<long>(unit_width)});
        }
        return result;
    }

    std::vector<Candidate> range_candidates(const sa::RangeSelectExpression& select,
                                            const slang::ConstantRange& range) {
        std::vector<Candidate> result;
        if (select.getSelectionKind() == sa::RangeSelectionKind::Simple) {
            const long a = translate(range, constant_int(select.left()));
            const long b = translate(range, constant_int(select.right()));
            result.push_back({kOne, std::min(a, b)});
            return result;
        }
        const long width = constant_int(select.right());
        const bool up = select.getSelectionKind() == sa::RangeSelectionKind::IndexedUp;
        const auto offset_for = [&](long base) {
            const long first = up ? base : base - width + 1;
            const long last = up ? base + width - 1 : base;
            return std::min(translate(range, first), translate(range, last));
        };
        const Word base = eval(select.left());
        if (synth::as_constant(base)) {
            result.push_back(
                {kOne, offset_for(signed_value(base, select.left().type->isSigned()))});
            return result;
        }
        for (long k = range.lower(); k <= range.upper(); ++k) {
            const Bit condition = ops_.equal(
                base, synth::constant_word(static_cast<std::uint64_t>(k), base.size()));
            if (condition == kZero) continue;
            result.push_back({condition, offset_for(k)});
        }
        return result;
    }

    Word gather(const Word& vector, const std::vector<Candidate>& candidates,
                std::size_t width) {
        Word result(width, kZero);
        for (const auto& candidate : candidates) {
            for (std::size_t j = 0; j < width; ++j) {
                const long position = candidate.offset + static_cast<long>(j);
                const Bit bit = position >= 0 && position < static_cast<long>(vector.size())
                                    ? vector[static_cast<std::size_t>(position)]
                                    : kZero;
                result[j] = candidate.condition == kOne
                                ? bit
                                : graph_.make_or(result[j],
                                                 graph_.make_and(candidate.condition, bit));
            }
        }
        return result;
    }

    Bit truthy(const Word& word) { return ops_.reduce_or(word); }

    Word apply_binary(sa::BinaryOperator op, const Word& left, const Word& right,
                      bool is_signed, std::size_t out_width) {
        using B = sa::BinaryOperator;
        const std::size_t common = std::max(left.size(), right.size());
        const auto one_bit = [&](Bit value) { return synth::resize({value}, out_width, false); };
        switch (op) {
            case B::Add:
                return ops_.add(synth::resize(left, out_width, is_signed),
                                synth::resize(right, out_width, is_signed));
            case B::Subtract:
                return ops_.sub(synth::resize(left, out_width, is_signed),
                                synth::resize(right, out_width, is_signed));
            case B::Multiply:
                return ops_.multiply(synth::resize(left, out_width, is_signed),
                                     synth::resize(right, out_width, is_signed));
            case B::BinaryAnd:
            case B::BinaryOr:
            case B::BinaryXor:
            case B::BinaryXnor: {
                const auto a = synth::resize(left, out_width, is_signed);
                const auto b = synth::resize(right, out_width, is_signed);
                if (op == B::BinaryAnd) return ops_.bitwise('&', a, b);
                if (op == B::BinaryOr) return ops_.bitwise('|', a, b);
                const auto x = ops_.bitwise('^', a, b);
                return op == B::BinaryXor ? x : ops_.bit_not(x);
            }
            case B::Equality:
            case B::CaseEquality:
                return one_bit(ops_.equal(synth::resize(left, common, is_signed),
                                          synth::resize(right, common, is_signed)));
            case B::Inequality:
            case B::CaseInequality:
                return one_bit(graph_.make_not(
                    ops_.equal(synth::resize(left, common, is_signed),
                               synth::resize(right, common, is_signed))));
            case B::LessThan:
            case B::LessThanEqual:
            case B::GreaterThan:
            case B::GreaterThanEqual: {
                const auto a = synth::resize(left, common, is_signed);
                const auto b = synth::resize(right, common, is_signed);
                if (op == B::LessThan) return one_bit(ops_.less_than(a, b, is_signed));
                if (op == B::GreaterThan) return one_bit(ops_.less_than(b, a, is_signed));
                if (op == B::LessThanEqual) {
                    return one_bit(graph_.make_not(ops_.less_than(b, a, is_signed)));
                }
                return one_bit(graph_.make_not(ops_.less_than(a, b, is_signed)));
            }
            case B::LogicalAnd:
                return one_bit(graph_.make_and(truthy(left), truthy(right)));
            case B::LogicalOr:
                return one_bit(graph_.make_or(truthy(left), truthy(right)));
            case B::LogicalImplication:
                return one_bit(graph_.make_or(graph_.make_not(truthy(left)), truthy(right)));
            case B::LogicalEquivalence:
                return one_bit(graph_.make_not(
                    graph_.make_xor(truthy(left), truthy(right))));
            case B::LogicalShiftLeft:
            case B::ArithmeticShiftLeft:
                return ops_.shift(synth::resize(left, out_width, is_signed), right,
                                  WordOps::Shift::Left);
            case B::LogicalShiftRight:
                return ops_.shift(synth::resize(left, out_width, false), right,
                                  WordOps::Shift::LogicalRight);
            case B::ArithmeticShiftRight:
                return ops_.shift(synth::resize(left, out_width, is_signed), right,
                                  is_signed ? WordOps::Shift::ArithmeticRight
                                            : WordOps::Shift::LogicalRight);
            case B::Divide:
            case B::Mod: {
                const auto a = synth::resize(left, out_width, is_signed);
                const auto b = synth::resize(right, out_width, is_signed);
                const auto result = ops_.divmod(a, b, is_signed);
                return op == B::Divide ? result.first : result.second;
            }
            case B::Power: {
                const auto base = synth::resize(left, out_width, is_signed);
                if (const auto exponent = synth::as_constant(right)) {
                    if (*exponent > 64) unsupported("constant exponent above 64");
                    Word result = synth::constant_word(1, out_width);
                    for (std::uint64_t i = 0; i < *exponent; ++i) {
                        result = ops_.multiply(result, base);
                    }
                    return result;
                }
                if (synth::as_constant(base) == std::optional<std::uint64_t>(2)) {
                    return ops_.shift(synth::constant_word(1, out_width), right,
                                      WordOps::Shift::Left);
                }
                unsupported("power with a non-constant exponent and a base other than 2");
            }
            default:
                unsupported("binary operator");
        }
    }

    Word eval(const sa::Expression& expr) {
        using K = sa::ExpressionKind;
        const std::size_t width = type_width(*expr.type);
        if (const auto constant = constant_of(expr)) return *constant;
        switch (expr.kind) {
            case K::HierarchicalValue:
            case K::NamedValue: {
                const auto& symbol = expr.as<sa::ValueExpressionBase>().symbol;
                Var& var = var_for(symbol);
                // Arrays are stored flat, so a whole-array read is the flat word.
                return synth::resize(read_var(var), width, false);
            }
            case K::Conversion: {
                const auto& conversion = expr.as<sa::ConversionExpression>();
                if (!expr.type->isIntegral()) unsupported("non-integral conversion");
                const auto& operand = conversion.operand();
                return synth::resize(eval(operand), width, operand.type->isSigned());
            }
            case K::UnaryOp: {
                const auto& unary = expr.as<sa::UnaryExpression>();
                const Word operand = eval(unary.operand());
                const bool is_signed = unary.operand().type->isSigned();
                using U = sa::UnaryOperator;
                const auto one_bit = [&](Bit value) {
                    return synth::resize({value}, width, false);
                };
                switch (unary.op) {
                    case U::Plus: return synth::resize(operand, width, is_signed);
                    case U::Minus:
                        return ops_.negate(synth::resize(operand, width, is_signed));
                    case U::BitwiseNot:
                        return ops_.bit_not(synth::resize(operand, width, is_signed));
                    case U::BitwiseAnd: return one_bit(ops_.reduce_and(operand));
                    case U::BitwiseOr: return one_bit(ops_.reduce_or(operand));
                    case U::BitwiseXor: return one_bit(ops_.reduce_xor(operand));
                    case U::BitwiseNand:
                        return one_bit(graph_.make_not(ops_.reduce_and(operand)));
                    case U::BitwiseNor:
                        return one_bit(graph_.make_not(ops_.reduce_or(operand)));
                    case U::BitwiseXnor:
                        return one_bit(graph_.make_not(ops_.reduce_xor(operand)));
                    case U::LogicalNot: return one_bit(graph_.make_not(truthy(operand)));
                    default: unsupported("increment/decrement inside an expression");
                }
            }
            case K::BinaryOp: {
                const auto& binary = expr.as<sa::BinaryExpression>();
                const bool is_signed = binary.left().type->isSigned() &&
                                       binary.right().type->isSigned();
                const bool shift = binary.op == sa::BinaryOperator::LogicalShiftLeft ||
                                   binary.op == sa::BinaryOperator::LogicalShiftRight ||
                                   binary.op == sa::BinaryOperator::ArithmeticShiftLeft ||
                                   binary.op == sa::BinaryOperator::ArithmeticShiftRight;
                const Word left = eval(binary.left());
                const Word right = eval(binary.right());
                return apply_binary(binary.op, left, right,
                                    shift ? binary.left().type->isSigned() : is_signed,
                                    width);
            }
            case K::ConditionalOp: {
                const auto& cond = expr.as<sa::ConditionalExpression>();
                if (cond.conditions.size() != 1 || cond.conditions[0].pattern) {
                    unsupported("pattern conditions in ?:");
                }
                const Bit select = truthy(eval(*cond.conditions[0].expr));
                const bool z_right = is_all_z(cond.right());
                if (z_right || is_all_z(cond.left())) {
                    const auto& data_expr = z_right ? cond.left() : cond.right();
                    Word data = synth::resize(eval(data_expr), width, data_expr.type->isSigned());
                    const Bit enable = z_right ? select : graph_.make_not(select);
                    for (auto& bit : data) bit = graph_.tri(bit, enable);
                    return data;
                }
                return ops_.mux(
                    select,
                    synth::resize(eval(cond.left()), width, cond.left().type->isSigned()),
                    synth::resize(eval(cond.right()), width, cond.right().type->isSigned()));
            }
            case K::Concatenation: {
                const auto& concat = expr.as<sa::ConcatenationExpression>();
                Word result;
                for (auto it = concat.operands().rbegin(); it != concat.operands().rend();
                     ++it) {
                    const Word part = eval(**it);
                    result.insert(result.end(), part.begin(), part.end());
                }
                return synth::resize(result, width, false);
            }
            case K::Replication: {
                const auto& rep = expr.as<sa::ReplicationExpression>();
                const long count = constant_int(rep.count());
                const Word part = eval(rep.concat());
                Word result;
                for (long i = 0; i < count; ++i) {
                    result.insert(result.end(), part.begin(), part.end());
                }
                return synth::resize(result, width, false);
            }
            case K::ElementSelect: {
                const auto& select = expr.as<sa::ElementSelectExpression>();
                const auto& value = select.value();
                if (value.type->isUnpackedArray()) {
                    if (value.kind != K::NamedValue) unsupported("nested array read");
                    Var& var = var_for(value.as<sa::NamedValueExpression>().symbol);
                    const auto candidates = element_candidates(
                        select.selector(), var.range, var.element_width, true);
                    return gather(read_var(var), candidates, width);
                }
                const Word vector = eval(value);
                const auto candidates = element_candidates(
                    select.selector(), value.type->getFixedRange(), width, false);
                return gather(vector, candidates, width);
            }
            case K::RangeSelect: {
                const auto& select = expr.as<sa::RangeSelectExpression>();
                const Word vector = eval(select.value());
                return gather(vector,
                              range_candidates(select, select.value().type->getFixedRange()),
                              width);
            }
            case K::SimpleAssignmentPattern:
                return pattern_word(
                    expr.as<sa::SimpleAssignmentPatternExpression>().elements(), *expr.type);
            case K::StructuredAssignmentPattern:
                return pattern_word(
                    expr.as<sa::StructuredAssignmentPatternExpression>().elements(),
                    *expr.type);
            case K::LValueReference:
                if (!lvalue_reference_) unsupported("lvalue reference outside an assignment");
                return synth::resize(eval(*lvalue_reference_), width, false);
            case K::Inside: {
                const auto& inside = expr.as<sa::InsideExpression>();
                const Word subject = eval(inside.left());
                bool is_signed = inside.left().type->isSigned();
                Bit match = kZero;
                for (const auto* item : inside.rangeList()) {
                    const auto compare_width = [&](const sa::Expression& e) {
                        return std::max(subject.size(), type_width(*e.type));
                    };
                    if (item->kind == K::ValueRange) {
                        const auto& range = item->as<sa::ValueRangeExpression>();
                        const bool item_signed =
                            is_signed && range.left().type->isSigned() &&
                            range.right().type->isSigned();
                        const std::size_t w = std::max(compare_width(range.left()),
                                                       compare_width(range.right()));
                        const Word v = synth::resize(subject, w, item_signed);
                        const Word lo = synth::resize(eval(range.left()), w, item_signed);
                        const Word hi = synth::resize(eval(range.right()), w, item_signed);
                        match = graph_.make_or(
                            match, graph_.make_and(
                                       graph_.make_not(ops_.less_than(v, lo, item_signed)),
                                       graph_.make_not(ops_.less_than(hi, v, item_signed))));
                    } else {
                        const bool item_signed = is_signed && item->type->isSigned();
                        const std::size_t w = compare_width(*item);
                        match = graph_.make_or(
                            match, ops_.equal(synth::resize(subject, w, item_signed),
                                              synth::resize(eval(*item), w, item_signed)));
                    }
                }
                return synth::resize(Word{match}, width, false);
            }
            case K::MemberAccess: {
                const auto& access = expr.as<sa::MemberAccessExpression>();
                if (!access.value().type->isIntegral() ||
                    access.member.kind != sa::SymbolKind::Field) {
                    unsupported("member access on a non-packed type");
                }
                const Word packed = eval(access.value());
                const auto offset = access.member.as<sa::FieldSymbol>().bitOffset;
                Word field(width, kZero);
                for (std::size_t i = 0; i < width && offset + i < packed.size(); ++i) {
                    field[i] = packed[offset + i];
                }
                return field;
            }
            case K::Call: {
                const auto& call = expr.as<sa::CallExpression>();
                if (call.isSystemCall()) {
                    const auto name = call.getSubroutineName();
                    if ((name == "$signed" || name == "$unsigned") &&
                        call.arguments().size() == 1) {
                        const auto& argument = *call.arguments()[0];
                        return synth::resize(eval(argument), width,
                                             argument.type->isSigned());
                    }
                }
                return synth::resize(call_subroutine(call), width,
                                     call.type->isSigned());
            }
            case K::Assignment:
                unsupported("assignment used as a value");
            default:
                unsupported("expression kind " + std::string(toString(expr.kind)));
        }
    }

    // Elements of an assignment pattern arrive in declaration order (struct fields,
    // or array elements from the left bound), already resolved against defaults.
    Word pattern_word(std::span<const sa::Expression* const> elements, const sa::Type& type) {
        const std::size_t total = type_width(type);
        Word result(total, kZero);
        const auto place = [&](const sa::Expression& element, std::size_t offset,
                               std::size_t width) {
            const Word value = synth::resize(eval(element), width, element.type->isSigned());
            for (std::size_t i = 0; i < width && offset + i < total; ++i) {
                result[offset + i] = value[i];
            }
        };
        const auto& canonical = type.getCanonicalType();
        if (canonical.isStruct() && canonical.isIntegral()) {
            std::size_t index = 0;
            for (const auto& field :
                 canonical.as<sa::PackedStructType>().membersOfType<sa::FieldSymbol>()) {
                if (index >= elements.size()) break;
                place(*elements[index++], field.bitOffset, type_width(field.getType()));
            }
            return result;
        }
        if (canonical.isUnpackedArray() || canonical.isIntegral()) {
            const auto* element_type = canonical.getArrayElementType();
            if (!element_type || !canonical.hasFixedRange() || elements.empty()) {
                unsupported("assignment pattern for this type");
            }
            const auto range = canonical.getFixedRange();
            const std::size_t element_width = type_width(*element_type);
            for (std::size_t i = 0; i < elements.size(); ++i) {
                const std::size_t position =
                    canonical.isUnpackedArray()
                        ? (range.left <= range.right ? i : elements.size() - 1 - i)
                        : elements.size() - 1 - i;
                place(*elements[i], position * element_width, element_width);
            }
            return result;
        }
        unsupported("assignment pattern for this type");
    }

    // ---- lvalues ----
    using TargetList = std::vector<std::vector<Target>>;

    TargetList lvalue_bits(const sa::Expression& expr) {
        using K = sa::ExpressionKind;
        const std::size_t width = type_width(*expr.type);
        switch (expr.kind) {
            case K::HierarchicalValue:
            case K::NamedValue: {
                Var& var = var_for(expr.as<sa::ValueExpressionBase>().symbol);
                TargetList result(var.width);
                for (std::size_t i = 0; i < var.width; ++i) result[i].push_back({&var, i, kOne});
                return result;
            }
            case K::ElementSelect: {
                const auto& select = expr.as<sa::ElementSelectExpression>();
                const auto& value = select.value();
                TargetList result(width);
                if (value.type->isUnpackedArray()) {
                    if (value.kind != K::NamedValue) unsupported("nested array write");
                    Var& var = var_for(value.as<sa::NamedValueExpression>().symbol);
                    for (const auto& candidate : element_candidates(
                             select.selector(), var.range, var.element_width, true)) {
                        if (candidate.offset < 0 ||
                            candidate.offset >= static_cast<long>(var.width)) {
                            continue;
                        }
                        for (std::size_t j = 0; j < width; ++j) {
                            result[j].push_back({&var,
                                                 static_cast<std::size_t>(candidate.offset) + j,
                                                 candidate.condition});
                        }
                    }
                    return result;
                }
                const auto base = lvalue_bits(value);
                for (const auto& candidate : element_candidates(
                         select.selector(), value.type->getFixedRange(), width, false)) {
                    add_slice(result, base, candidate);
                }
                return result;
            }
            case K::RangeSelect: {
                const auto& select = expr.as<sa::RangeSelectExpression>();
                TargetList result(width);
                const auto base = lvalue_bits(select.value());
                for (const auto& candidate : range_candidates(
                         select, select.value().type->getFixedRange())) {
                    add_slice(result, base, candidate);
                }
                return result;
            }
            case K::MemberAccess: {
                const auto& access = expr.as<sa::MemberAccessExpression>();
                if (!access.value().type->isIntegral() ||
                    access.member.kind != sa::SymbolKind::Field) {
                    unsupported("member access on a non-packed type");
                }
                const auto base = lvalue_bits(access.value());
                const auto offset = access.member.as<sa::FieldSymbol>().bitOffset;
                TargetList result(width);
                for (std::size_t j = 0; j < width && offset + j < base.size(); ++j) {
                    result[j] = base[offset + j];
                }
                return result;
            }
            case K::Concatenation: {
                TargetList result;
                const auto& concat = expr.as<sa::ConcatenationExpression>();
                for (auto it = concat.operands().rbegin(); it != concat.operands().rend();
                     ++it) {
                    auto part = lvalue_bits(**it);
                    result.insert(result.end(), part.begin(), part.end());
                }
                return result;
            }
            default:
                unsupported("assignment target (expression kind " +
                            std::string(toString(expr.kind)) + ")");
        }
    }

    void add_slice(TargetList& result, const TargetList& base, const Candidate& candidate) {
        for (std::size_t j = 0; j < result.size(); ++j) {
            const long position = candidate.offset + static_cast<long>(j);
            if (position < 0 || position >= static_cast<long>(base.size())) continue;
            for (const auto& target : base[static_cast<std::size_t>(position)]) {
                result[j].push_back(
                    {target.var, target.bit,
                     graph_.make_and(candidate.condition, target.condition)});
            }
        }
    }

    Word& touch_cur(Var& var) {
        auto [it, inserted] = st_.cur.try_emplace(var.symbol);
        if (inserted) {
            it->second = seq_ && var.registered ? var.wires : Word(var.width, kLatch);
        }
        return it->second;
    }
    Word& touch_nb(Var& var) {
        auto [it, inserted] = st_.nb.try_emplace(var.symbol);
        if (inserted) it->second = var.wires;
        return it->second;
    }

    void write(const sa::Expression& lhs, Word value, bool nonblocking) {
        const auto targets = lvalue_bits(lhs);
        value = synth::resize(value, targets.size(), false);
        for (std::size_t p = 0; p < targets.size(); ++p) {
            for (const auto& target : targets[p]) {
                Word& destination =
                    nonblocking && seq_ ? touch_nb(*target.var) : touch_cur(*target.var);
                destination[target.bit] =
                    target.condition == kOne
                        ? value[p]
                        : graph_.make_mux(target.condition, value[p], destination[target.bit]);
            }
        }
    }

    // ---- statements ----
    Word initial_value(const Var& var, bool nonblocking) const {
        if (nonblocking) return var.wires;
        return seq_ && var.registered ? var.wires : Word(var.width, kLatch);
    }

    void merge_maps(Bit condition,
                    std::unordered_map<const sa::Symbol*, Word>& when_true,
                    std::unordered_map<const sa::Symbol*, Word>& when_false,
                    std::unordered_map<const sa::Symbol*, Word>& out, bool nonblocking) {
        std::set<const sa::Symbol*> keys;
        for (const auto& entry : when_true) keys.insert(entry.first);
        for (const auto& entry : when_false) keys.insert(entry.first);
        for (const auto* key : keys) {
            const Var& var = vars_.at(key);
            const auto t = when_true.find(key);
            const auto f = when_false.find(key);
            const Word tw = t != when_true.end() ? t->second : initial_value(var, nonblocking);
            const Word fw = f != when_false.end() ? f->second : initial_value(var, nonblocking);
            Word merged(tw.size());
            for (std::size_t i = 0; i < tw.size(); ++i) {
                merged[i] = graph_.make_mux(condition, tw[i], fw[i]);
            }
            out[key] = std::move(merged);
        }
    }

    State merge(Bit condition, State& when_true, State& when_false) {
        State result;
        merge_maps(condition, when_true.cur, when_false.cur, result.cur, false);
        merge_maps(condition, when_true.nb, when_false.nb, result.nb, true);
        result.returned = graph_.make_mux(condition, when_true.returned, when_false.returned);
        return result;
    }

    void exec_branch(Bit condition, const std::function<void()>& then_branch,
                     const std::function<void()>& else_branch) {
        if (condition == kOne) {
            then_branch();
            return;
        }
        if (condition == kZero) {
            else_branch();
            return;
        }
        State saved = st_;
        then_branch();
        State taken = std::move(st_);
        st_ = std::move(saved);
        else_branch();
        st_ = merge(condition, taken, st_);
    }

    Bit case_match(const sa::CaseStatement& statement, const Word& subject,
                   const sa::Expression& item) {
        const bool is_signed = statement.expr.type->isSigned() && item.type->isSigned();
        const std::size_t width = std::max(subject.size(), type_width(*item.type));
        const Word a = synth::resize(subject, width, is_signed);
        const Word b = synth::resize(eval(item), width, is_signed);
        if (statement.condition == sa::CaseStatementCondition::Normal) {
            return ops_.equal(a, b);
        }
        std::vector<char> ignore(width, 0);
        sa::EvalContext context(*anchor_);
        const auto constant = item.eval(context);
        if (constant.isInteger()) {
            const auto& integer = constant.integer();
            for (std::size_t i = 0; i < width && i < integer.getBitWidth(); ++i) {
                const auto bit = integer[static_cast<int32_t>(i)];
                if (!bit.isUnknown()) continue;
                const bool is_x = bit.value == slang::logic_t::x.value;
                if (is_x && statement.condition == sa::CaseStatementCondition::WildcardJustZ) {
                    continue;
                }
                ignore[i] = 1;
            }
        }
        Bit result = kOne;
        for (std::size_t i = 0; i < width; ++i) {
            if (ignore[i]) continue;
            result = graph_.make_and(result,
                                     graph_.make_not(graph_.make_xor(a[i], b[i])));
        }
        return result;
    }

    bool covers_all(const std::vector<std::pair<Bit, State>>& branches) {
        Bit covered = kZero;
        for (const auto& branch : branches) covered = graph_.make_or(covered, branch.first);
        return is_tautology(covered);
    }

    // Exhaustive check over the (small) support of a condition.
    bool is_tautology(Bit root) {
        if (root == kOne) return true;
        std::vector<Bit> leaves;
        std::unordered_set<Bit> seen;
        std::vector<Bit> stack{root};
        while (!stack.empty()) {
            const Bit id = stack.back();
            stack.pop_back();
            if (!seen.insert(id).second) continue;
            const Node& n = graph_.node(id);
            switch (n.op) {
                case Op::Const: break;
                case Op::Wire:
                    if (n.a >= 0) stack.push_back(n.a);
                    else leaves.push_back(id);
                    break;
                case Op::Not: stack.push_back(n.a); break;
                case Op::And:
                case Op::Or:
                case Op::Xor: stack.push_back(n.a); stack.push_back(n.b); break;
                case Op::Mux: stack.push_back(n.a); stack.push_back(n.b); stack.push_back(n.c); break;
                default: leaves.push_back(id); break;
            }
            if (leaves.size() > 14) return false;
        }
        std::unordered_map<Bit, bool> value;
        std::function<bool(Bit)> evaluate = [&](Bit id) -> bool {
            const Node& n = graph_.node(id);
            switch (n.op) {
                case Op::Const: return id == kOne;
                case Op::Wire: return n.a >= 0 ? evaluate(n.a) : value.at(id);
                case Op::Not: return !evaluate(n.a);
                case Op::And: return evaluate(n.a) && evaluate(n.b);
                case Op::Or: return evaluate(n.a) || evaluate(n.b);
                case Op::Xor: return evaluate(n.a) != evaluate(n.b);
                case Op::Mux: return evaluate(n.c) ? evaluate(n.b) : evaluate(n.a);
                default: return value.at(id);
            }
        };
        for (std::uint32_t mask = 0; mask < (1u << leaves.size()); ++mask) {
            for (std::size_t i = 0; i < leaves.size(); ++i) value[leaves[i]] = (mask >> i) & 1u;
            if (!evaluate(root)) return false;
        }
        return true;
    }

    bool has_full_case(const sa::CaseStatement& node) const {
        for (const auto* attribute : anchor_->getParentScope()->getCompilation().getAttributes(node)) {
            if (attribute->name == "full_case") return true;
        }
        return false;
    }

    static bool is_all_z(const sa::Expression& expr) {
        const sa::Expression* current = &expr;
        while (current->kind == sa::ExpressionKind::Conversion) {
            current = &current->as<sa::ConversionExpression>().operand();
        }
        if (current->kind != sa::ExpressionKind::IntegerLiteral) return false;
        const auto& value = current->as<sa::IntegerLiteral>().getValue();
        for (slang::bitwidth_t i = 0; i < value.getBitWidth(); ++i) {
            if (value[i].value != slang::logic_t::Z_VALUE) return false;
        }
        return value.getBitWidth() > 0;
    }

    // Inlines a user function/task call; returns the function result (empty for tasks).
    Word call_subroutine(const sa::CallExpression& call) {
        const auto* subroutine = std::get<0>(call.subroutine);
        const auto formals = subroutine->getArguments();
        const auto actuals = call.arguments();
        if (++call_depth_ > 32) unsupported("recursive function '" + std::string(subroutine->name) + "'");
        std::vector<Word> inputs(formals.size());
        for (std::size_t i = 0; i < formals.size(); ++i) {
            const auto direction = formals[i]->direction;
            if (direction == sa::ArgumentDirection::Out) continue;
            if (direction == sa::ArgumentDirection::Ref) unsupported("ref arguments");
            inputs[i] = synth::resize(eval(*actuals[i]), type_width(formals[i]->getType()),
                                      actuals[i]->type->isSigned());
        }
        for (std::size_t i = 0; i < formals.size(); ++i) {
            Var& formal = var_for(*formals[i]);
            st_.cur[formals[i]] = formals[i]->direction == sa::ArgumentDirection::Out
                                      ? Word(formal.width, kLatch)
                                      : inputs[i];
        }
        Var* result_var = nullptr;
        if (subroutine->returnValVar) {
            result_var = &var_for(*subroutine->returnValVar);
            st_.cur[subroutine->returnValVar] = Word(result_var->width, kLatch);
        }
        Var* const outer_return_var = return_var_;
        const Bit outer_returned = st_.returned;
        return_var_ = result_var;
        st_.returned = kZero;
        exec(subroutine->getBody());
        return_var_ = outer_return_var;
        st_.returned = outer_returned;
        for (std::size_t i = 0; i < formals.size(); ++i) {
            const auto direction = formals[i]->direction;
            if (direction != sa::ArgumentDirection::Out &&
                direction != sa::ArgumentDirection::InOut) {
                continue;
            }
            const sa::Expression* target = actuals[i];
            if (target->kind == sa::ExpressionKind::Assignment) {
                target = &target->as<sa::AssignmentExpression>().left();
            }
            write(*target, st_.cur.at(formals[i]), false);
        }
        --call_depth_;
        return result_var ? st_.cur.at(subroutine->returnValVar) : Word{};
    }

    void exec_expression(const sa::Expression& expr) {
        using K = sa::ExpressionKind;
        if (expr.kind == K::Assignment) {
            const auto& assignment = expr.as<sa::AssignmentExpression>();
            const auto& left = assignment.left();
            // For compound assignments slang folds the operator into right(), with an
            // LValueReference standing for the current value of the target.
            const sa::Expression* const outer_reference = lvalue_reference_;
            lvalue_reference_ = &left;
            Word value = eval(assignment.right());
            lvalue_reference_ = outer_reference;
            const std::size_t width = type_width(*left.type);
            value = synth::resize(value, width, assignment.right().type->isSigned());
            write(left, value, assignment.isNonBlocking());
            return;
        }
        if (expr.kind == K::UnaryOp) {
            const auto& unary = expr.as<sa::UnaryExpression>();
            using U = sa::UnaryOperator;
            if (unary.op == U::Preincrement || unary.op == U::Postincrement ||
                unary.op == U::Predecrement || unary.op == U::Postdecrement) {
                const std::size_t width = type_width(*unary.operand().type);
                const Word one = synth::constant_word(1, width);
                const Word current = eval(unary.operand());
                const bool increment =
                    unary.op == U::Preincrement || unary.op == U::Postincrement;
                write(unary.operand(), increment ? ops_.add(current, one)
                                                 : ops_.sub(current, one),
                      false);
                return;
            }
        }
        if (expr.kind == K::Call && expr.as<sa::CallExpression>().isSystemCall()) {
            return;  // $display and friends have no hardware meaning
        }
        if (expr.kind == K::Call) {
            call_subroutine(expr.as<sa::CallExpression>());
            return;
        }
        unsupported("expression statement");
    }

    // Statements after a possible `return` only run when it has not fired.
    void exec_sequence(std::span<const sa::Statement* const> list, std::size_t first) {
        for (std::size_t i = first; i < list.size(); ++i) {
            exec(*list[i]);
            if (st_.returned == kZero || i + 1 == list.size()) continue;
            if (st_.returned == kOne) return;
            exec_branch(graph_.make_not(st_.returned), [&] { exec_sequence(list, i + 1); },
                        [] {});
            return;
        }
    }

    void exec(const sa::Statement& statement) {
        using S = sa::StatementKind;
        switch (statement.kind) {
            case S::Empty:
            case S::ImmediateAssertion:
            case S::ConcurrentAssertion:
                return;
            case S::List:
                exec_sequence(statement.as<sa::StatementList>().list, 0);
                return;
            case S::Return: {
                const auto& node = statement.as<sa::ReturnStatement>();
                if (node.expr && return_var_ != nullptr) {
                    touch_cur(*return_var_) =
                        synth::resize(eval(*node.expr), return_var_->width,
                                      node.expr->type->isSigned());
                }
                st_.returned = kOne;
                return;
            }
            case S::Block:
                exec(statement.as<sa::BlockStatement>().body);
                return;
            case S::Timed:
                exec(statement.as<sa::TimedStatement>().stmt);
                return;
            case S::VariableDeclaration: {
                const auto& symbol = statement.as<sa::VariableDeclStatement>().symbol;
                if (const auto* initializer = symbol.getInitializer()) {
                    Var& var = var_for(symbol);
                    touch_cur(var) = synth::resize(eval(*initializer), var.width,
                                                   initializer->type->isSigned());
                }
                return;
            }
            case S::ExpressionStatement:
                exec_expression(statement.as<sa::ExpressionStatement>().expr);
                return;
            case S::Conditional: {
                const auto& node = statement.as<sa::ConditionalStatement>();
                Bit condition = kOne;
                for (const auto& entry : node.conditions) {
                    if (entry.pattern) unsupported("pattern-matching if");
                    condition = graph_.make_and(condition, truthy(eval(*entry.expr)));
                }
                exec_branch(
                    condition, [&] { exec(node.ifTrue); },
                    [&] {
                        if (node.ifFalse) exec(*node.ifFalse);
                    });
                return;
            }
            case S::Case: {
                const auto& node = statement.as<sa::CaseStatement>();
                const Word subject = eval(node.expr);
                State base = st_;
                std::vector<std::pair<Bit, State>> branches;
                for (const auto& group : node.items) {
                    Bit match = kZero;
                    for (const auto* item : group.expressions) {
                        match = graph_.make_or(match, case_match(node, subject, *item));
                    }
                    st_ = base;
                    exec(*group.stmt);
                    branches.emplace_back(match, std::move(st_));
                }
                st_ = std::move(base);
                if (node.defaultCase) {
                    exec(*node.defaultCase);
                } else if (!branches.empty() && (has_full_case(node) || covers_all(branches))) {
                    // full_case: uncovered values are don't-care, so reuse the last item
                    st_ = std::move(branches.back().second);
                    branches.back().first = kZero;
                }
                for (auto it = branches.rbegin(); it != branches.rend(); ++it) {
                    if (it->first == kOne) {
                        st_ = std::move(it->second);
                    } else if (it->first != kZero) {
                        st_ = merge(it->first, it->second, st_);
                    }
                }
                return;
            }
            case S::ForLoop: {
                const auto& loop = statement.as<sa::ForLoopStatement>();
                for (const auto* variable : loop.loopVars) {
                    if (const auto* initializer = variable->getInitializer()) {
                        Var& var = var_for(*variable);
                        touch_cur(var) = synth::resize(eval(*initializer), var.width,
                                                       initializer->type->isSigned());
                    }
                }
                for (const auto* initializer : loop.initializers) exec_expression(*initializer);
                for (int iteration = 0;; ++iteration) {
                    if (iteration > 65536) unsupported("loop with more than 65536 iterations");
                    if (loop.stopExpr) {
                        const Bit condition = truthy(eval(*loop.stopExpr));
                        if (condition == kZero) break;
                        if (condition != kOne) unsupported("loop with a non-constant bound");
                    }
                    exec(loop.body);
                    for (const auto* step : loop.steps) exec_expression(*step);
                }
                return;
            }
            default:
                unsupported("statement kind " + std::string(toString(statement.kind)));
        }
    }

    // ---- process completion ----
    void begin_process(bool seq) {
        st_ = State{};
        seq_ = seq;
    }

    // ---- latch inference helpers ----
    bool has_latch(Bit id, std::unordered_map<Bit, bool>& memo) {
        if (id == kLatch) return true;
        const auto found = memo.find(id);
        if (found != memo.end()) return found->second;
        const Node n = graph_.node(id);
        bool result = false;
        switch (n.op) {
            case Op::Not: result = has_latch(n.a, memo); break;
            case Op::And:
            case Op::Or:
            case Op::Xor: result = has_latch(n.a, memo) || has_latch(n.b, memo); break;
            case Op::Mux:
                result = has_latch(n.a, memo) || has_latch(n.b, memo) || has_latch(n.c, memo);
                break;
            default: break;
        }
        memo[id] = result;
        return result;
    }

    // Data value with unassigned paths replaced by 0, and the condition under which
    // the signal is assigned (the latch enable).
    std::pair<Bit, Bit> latch_parts(Bit id, std::unordered_map<Bit, std::pair<Bit, Bit>>& memo) {
        if (id == kLatch) return {kZero, kZero};
        const auto found = memo.find(id);
        if (found != memo.end()) return found->second;
        const Node n = graph_.node(id);
        std::pair<Bit, Bit> result{id, kOne};
        switch (n.op) {
            case Op::Not: {
                const auto x = latch_parts(n.a, memo);
                result = {graph_.make_not(x.first), x.second};
                break;
            }
            case Op::And:
            case Op::Or:
            case Op::Xor: {
                const auto x = latch_parts(n.a, memo);
                const auto y = latch_parts(n.b, memo);
                const Bit data = n.op == Op::And   ? graph_.make_and(x.first, y.first)
                                 : n.op == Op::Or ? graph_.make_or(x.first, y.first)
                                                  : graph_.make_xor(x.first, y.first);
                result = {data, graph_.make_and(x.second, y.second)};
                break;
            }
            case Op::Mux: {
                const auto sel = latch_parts(n.c, memo);
                const auto hi = latch_parts(n.b, memo);
                const auto lo = latch_parts(n.a, memo);
                result = {graph_.make_mux(sel.first, hi.first, lo.first),
                          graph_.make_and(sel.second,
                                          graph_.make_mux(sel.first, hi.second, lo.second))};
                break;
            }
            default: break;
        }
        memo[id] = result;
        return result;
    }

    void finish_combinational() {
        std::unordered_map<Bit, bool> latch_memo;
        std::unordered_map<Bit, std::pair<Bit, Bit>> parts_memo;
        for (auto& [symbol, word] : st_.cur) {
            Var& var = vars_.at(symbol);
            if (!var.registered) continue;
            for (std::size_t i = 0; i < var.width; ++i) {
                Bit value = word[i];
                if (value == kLatch) continue;
                if (has_latch(value, latch_memo)) {
                    const auto [data, enable] = latch_parts(value, parts_memo);
                    if (is_tautology(enable)) {
                        value = data;
                    } else {
                        value = graph_.latch_cell(bit_name(var, i));
                        graph_.node(value).a = data;
                        graph_.node(value).b = enable;
                    }
                }
                drive(var, i, value);
            }
        }
    }

    struct AsyncLevel {
        Bit active;  // asserted while this asynchronous control is active
        State state;
    };

    std::optional<Bit> level_value(const AsyncLevel& level, const sa::Symbol* key,
                                   std::size_t bit, const Var& var) {
        const auto nb = level.state.nb.find(key);
        if (nb != level.state.nb.end() && nb->second[bit] != var.wires[bit]) {
            return nb->second[bit];
        }
        const auto cur = level.state.cur.find(key);
        if (cur != level.state.cur.end() && cur->second[bit] != var.wires[bit]) {
            return cur->second[bit];
        }
        return std::nullopt;
    }

    void finish_sequential(Bit clock, bool negedge, const std::vector<AsyncLevel>& levels,
                           bool both_edges = false) {
        std::set<const sa::Symbol*> keys;
        for (const auto& entry : st_.cur) keys.insert(entry.first);
        for (const auto& entry : st_.nb) keys.insert(entry.first);
        for (const auto& level : levels) {
            for (const auto& entry : level.state.cur) keys.insert(entry.first);
            for (const auto& entry : level.state.nb) keys.insert(entry.first);
        }
        for (const auto* key : keys) {
            Var& var = vars_.at(key);
            if (!var.registered) continue;
            const auto cur = st_.cur.find(key);
            const auto nb = st_.nb.find(key);
            for (std::size_t i = 0; i < var.width; ++i) {
                Bit d = cur != st_.cur.end() ? cur->second[i] : var.wires[i];
                if (nb != st_.nb.end() && nb->second[i] != var.wires[i]) d = nb->second[i];
                Bit act[2] = {kZero, kZero};  // asynchronous clear / set conditions
                bool used[2] = {false, false};
                Bit earlier = kZero;  // controls listed first take priority
                for (const auto& level : levels) {
                    const auto value = level_value(level, key, i, var);
                    if (!value) {
                        // The control freezes this bit while it is asserted.
                        d = graph_.make_mux(level.active, var.wires[i], d);
                        continue;
                    }
                    if (*value != kZero && *value != kOne) {
                        unsupported("non-constant asynchronous reset value for " +
                                    bit_name(var, i));
                    }
                    const int kind = *value == kOne ? 1 : 0;
                    used[kind] = true;
                    act[kind] = graph_.make_or(
                        act[kind], graph_.make_and(level.active, graph_.make_not(earlier)));
                    earlier = graph_.make_or(earlier, level.active);
                }
                if (!used[0] && !used[1] && (d == var.wires[i] || d == kLatch)) continue;
                if (d == kLatch) d = var.wires[i];
                if (both_edges) {
                    if (used[0] || used[1]) unsupported("asynchronous reset on a dual-edge flop");
                    // q = p ^ n: p captures (d ^ n) on rising edges, n captures (d ^ p) on
                    // falling edges, so q equals d after either edge.
                    const Bit p = graph_.flop(bit_name(var, i) + "_pos", false);
                    const Bit n = graph_.flop(bit_name(var, i) + "_neg", true);
                    graph_.node(p).a = graph_.make_xor(d, n);
                    graph_.node(p).b = clock;
                    graph_.node(n).a = graph_.make_xor(d, p);
                    graph_.node(n).b = clock;
                    drive(var, i, graph_.make_xor(p, n));
                    continue;
                }
                const Bit q = graph_.flop(bit_name(var, i), negedge);
                graph_.node(q).a = d;
                graph_.node(q).b = clock;
                if (used[0] && used[1]) {
                    graph_.node(q).c = act[0];
                    graph_.node(q).d = act[1];
                } else if (used[0] || used[1]) {
                    graph_.node(q).c = used[1] ? act[1] : act[0];
                    graph_.node(q).reset_value = used[1];
                }
                drive(var, i, q);
            }
        }
    }

    static const sa::Statement& unwrap_single(const sa::Statement& statement) {
        const sa::Statement* current = &statement;
        for (;;) {
            if (current->kind == sa::StatementKind::Block) {
                current = &current->as<sa::BlockStatement>().body;
            } else if (current->kind == sa::StatementKind::List &&
                       current->as<sa::StatementList>().list.size() == 1) {
                current = current->as<sa::StatementList>().list[0];
            } else {
                return *current;
            }
        }
    }

    static const sa::Symbol* control_symbol(const sa::Expression* expr) {
        while (expr) {
            if (expr->kind == sa::ExpressionKind::UnaryOp) {
                expr = &expr->as<sa::UnaryExpression>().operand();
            } else if (expr->kind == sa::ExpressionKind::Conversion) {
                expr = &expr->as<sa::ConversionExpression>().operand();
            } else if (expr->kind == sa::ExpressionKind::NamedValue) {
                return &expr->as<sa::NamedValueExpression>().symbol;
            } else {
                return nullptr;
            }
        }
        return nullptr;
    }

    void run_procedure(const sa::ProceduralBlockSymbol& block) {
        using P = sa::ProceduralBlockKind;
        if (block.procedureKind == P::Initial || block.procedureKind == P::Final) return;
        const sa::Statement* body = &block.getBody();
        const sa::TimingControl* timing = nullptr;
        if (body->kind == sa::StatementKind::Timed) {
            timing = &body->as<sa::TimedStatement>().timing;
            body = &body->as<sa::TimedStatement>().stmt;
        }
        struct Event {
            const sa::Expression* expr;
            bool negedge;
            bool both;
        };
        std::vector<Event> events;
        const auto consider = [&](const sa::TimingControl& control) {
            if (control.kind != sa::TimingControlKind::SignalEvent) return;
            const auto& event = control.as<sa::SignalEventControl>();
            if (event.edge == sa::EdgeKind::None) return;
            if (event.edge == sa::EdgeKind::BothEdges) {
                events.push_back({&event.expr, false, true});
                return;
            }
            events.push_back({&event.expr, event.edge == sa::EdgeKind::NegEdge, false});
        };
        if (timing) {
            if (timing->kind == sa::TimingControlKind::EventList) {
                for (const auto* event : timing->as<sa::EventListControl>().events) {
                    consider(*event);
                }
            } else {
                consider(*timing);
            }
        }
        // "posedge x or negedge x" is a dual-edge clock.
        for (std::size_t x = 0; x < events.size(); ++x) {
            for (std::size_t y = x + 1; y < events.size(); ++y) {
                const auto* sx = control_symbol(events[x].expr);
                if (sx && sx == control_symbol(events[y].expr) &&
                    events[x].negedge != events[y].negedge && !events[x].both &&
                    !events[y].both) {
                    events[x].both = true;
                    events.erase(events.begin() + static_cast<std::ptrdiff_t>(y));
                    break;
                }
            }
        }
        const bool sequential = !events.empty();
        if (block.procedureKind == P::AlwaysFF && !sequential) {
            unsupported("always_ff without a clock edge");
        }
        begin_process(sequential);
        std::vector<AsyncLevel> levels;
        Event clock_event{nullptr, false, false};
        if (sequential) {
            // Edge events whose signal is tested by the leading if/else-if chain are
            // asynchronous set/reset controls; the remaining one is the clock.
            std::vector<char> is_control(events.size(), 0);
            std::size_t controls = 0;
            while (events.size() - controls > 1) {
                const auto& statement = unwrap_single(*body);
                if (statement.kind != sa::StatementKind::Conditional) {
                    unsupported("asynchronous control without a leading if");
                }
                const auto& node = statement.as<sa::ConditionalStatement>();
                if (node.conditions.size() != 1 || node.conditions[0].pattern || !node.ifFalse) {
                    unsupported("asynchronous reset structure");
                }
                const auto* symbol = control_symbol(node.conditions[0].expr);
                std::size_t match = events.size();
                for (std::size_t e = 0; e < events.size(); ++e) {
                    if (!is_control[e] && control_symbol(events[e].expr) == symbol) match = e;
                }
                if (!symbol || match == events.size()) {
                    unsupported("asynchronous control not matching the sensitivity list");
                }
                is_control[match] = 1;
                ++controls;
                const Bit active = truthy(eval(*node.conditions[0].expr));
                State saved = std::move(st_);
                st_ = State{};
                exec(node.ifTrue);
                levels.push_back({active, std::move(st_)});
                st_ = std::move(saved);
                body = node.ifFalse;
            }
            for (std::size_t e = 0; e < events.size(); ++e) {
                if (!is_control[e]) clock_event = events[e];
            }
        }
        Bit clock = kZero;
        if (sequential) {
            const Word word = eval(*clock_event.expr);
            clock = word.empty() ? kZero : word.front();
        }
        exec(*body);
        if (sequential) {
            finish_sequential(clock, clock_event.negedge, levels, clock_event.both);
        } else {
            finish_combinational();
        }
    }

    void run_assign(const sa::Expression& lhs, const sa::Expression& rhs) {
        begin_process(false);
        write(lhs, synth::resize(eval(rhs), type_width(*lhs.type), rhs.type->isSigned()),
              false);
        finish_combinational();
    }

    void link_instance(const sa::InstanceSymbol& child) {
        for (const auto* connection : child.getPortConnections()) {
            if (connection->port.kind != sa::SymbolKind::Port) continue;
            const auto& port = connection->port.as<sa::PortSymbol>();
            const auto* expression = connection->getExpression();
            if (!expression || !port.internalSymbol) continue;
            if (port.internalSymbol->kind != sa::SymbolKind::Net &&
                port.internalSymbol->kind != sa::SymbolKind::Variable) {
                unsupported("port '" + std::string(port.name) + "' of non-net type");
            }
            Var& inner = var_for(port.internalSymbol->as<sa::ValueSymbol>());
            if (port.direction == sa::ArgumentDirection::In) {
                begin_process(false);
                const Word value = synth::resize(eval(*expression), inner.width,
                                                 expression->type->isSigned());
                for (std::size_t i = 0; i < inner.width; ++i) drive(inner, i, value[i]);
            } else if (port.direction == sa::ArgumentDirection::Out) {
                const sa::Expression* target = expression;
                if (target->kind == sa::ExpressionKind::Assignment) {
                    target = &target->as<sa::AssignmentExpression>().left();
                }
                begin_process(false);
                write(*target, inner.wires, false);
                finish_combinational();
            } else {
                unsupported("inout port '" + std::string(port.name) + "'");
            }
        }
    }

    // ---- netlist emission ----
    struct Emitter;
};

struct Rebuilder {
    BitGraph& source;
    BitGraph target;
    std::vector<Bit> memo;
    std::vector<char> state;
    std::vector<Bit> pending;
    std::string context;
    std::vector<std::string> trail;

    explicit Rebuilder(BitGraph& graph)
        : source(graph), memo(graph.size(), -1), state(graph.size(), 0) {}

    Bit visit(Bit id) {
        if (memo[static_cast<std::size_t>(id)] != -1) return memo[static_cast<std::size_t>(id)];
        const Node& n = source.node(id);
        Bit result = kZero;
        switch (n.op) {
            case Op::Const: result = id; break;
            case Op::Latch:
                throw std::runtime_error(
                    "behavioral synthesis: latch inferred (signal not assigned on every "
                    "path) affecting " + context +
                    " (unassigned path through " +
                    (trail.empty() ? std::string("?") : trail.back()) + ")");
            case Op::Input: result = target.input(n.name); break;
            case Op::Wire: {
                auto& mark = state[static_cast<std::size_t>(id)];
                if (mark == 1) {
                    throw std::runtime_error(
                        "behavioral synthesis: combinational loop detected near " + context);
                }
                mark = 1;
                trail.push_back(n.name);
                result = n.a < 0 ? kZero : visit(n.a);
                trail.pop_back();
                mark = 2;
                break;
            }
            case Op::Not: result = target.make_not(visit(n.a)); break;
            case Op::And: result = target.make_and(visit(n.a), visit(n.b)); break;
            case Op::Or: result = target.make_or(visit(n.a), visit(n.b)); break;
            case Op::Xor: result = target.make_xor(visit(n.a), visit(n.b)); break;
            case Op::Mux: {
                const Bit select = visit(n.c);
                const Bit when_true = visit(n.b);
                const Bit when_false = visit(n.a);
                result = target.make_mux(select, when_true, when_false);
                break;
            }
            case Op::Flop:
                result = target.flop(n.name, n.negedge);
                target.node(result).reset_value = n.reset_value;
                memo[static_cast<std::size_t>(id)] = result;
                pending.push_back(id);
                return result;
            case Op::Tri: result = target.tri(visit(n.a), target.make_not(visit(n.b))); break;  // b becomes active-low
            case Op::LatchCell:
                result = target.latch_cell(n.name);
                memo[static_cast<std::size_t>(id)] = result;
                pending.push_back(id);
                return result;
        }
        memo[static_cast<std::size_t>(id)] = result;
        return result;
    }

    void drain() {
        while (!pending.empty()) {
            const Bit id = pending.back();
            pending.pop_back();
            const Node& flop = source.node(id);
            context = flop.name;
            const Bit data = visit(flop.a);
            const Bit clock = visit(flop.b);
            const Bit reset = flop.c >= 0 ? target.make_not(visit(flop.c)) : -1;
            const Bit set = flop.d >= 0 ? target.make_not(visit(flop.d)) : -1;
            auto& created = target.node(memo[static_cast<std::size_t>(id)]);
            created.a = data;
            created.b = clock;
            created.c = reset;
            created.d = set;
        }
    }
};

Module Synthesizer::run(const sa::InstanceSymbol& top, const std::string& top_name) {
    top_prefix_ = top_name + ".";
    anchor_ = &top.body;
    collect(top.body);

    std::vector<Port> ports;
    std::vector<Net> nets;
    std::vector<Output> outputs;
    std::map<Bit, std::string> input_names;
    for (const auto* port_symbol : top.body.getPortList()) {
        if (port_symbol->kind != sa::SymbolKind::Port) continue;
        const auto& port = port_symbol->as<sa::PortSymbol>();
        if (!port.internalSymbol) continue;
        Var& var = var_for(port.internalSymbol->as<sa::ValueSymbol>());
        if (port.direction == sa::ArgumentDirection::InOut) {
            unsupported("inout port '" + std::string(port.name) + "'");
        }
        const bool is_input = port.direction == sa::ArgumentDirection::In;
        ports.emplace_back(var.name, is_input ? PortDirection::Input : PortDirection::Output);
        for (std::size_t i = 0; i < var.width; ++i) {
            const std::string name = bit_name(var, i);
            nets.emplace_back(name, is_input ? NetType::Input : NetType::Output);
            if (is_input) {
                drive(var, i, graph_.input(name));
            } else {
                outputs.push_back({name, &var, i});
            }
        }
    }

    for (const auto* child : instances_) link_instance(*child);
    for (const auto* net : net_initializers_) {
        Var& var = var_for(*net);
        begin_process(false);
        Word& slot = touch_cur(var);
        slot = synth::resize(eval(*net->getInitializer()), var.width,
                             net->getInitializer()->type->isSigned());
        finish_combinational();
    }
    for (const auto* assign : assigns_) {
        const auto& expression = assign->getAssignment().as<sa::AssignmentExpression>();
        run_assign(expression.left(), expression.right());
    }
    for (const auto* block : procedures_) run_procedure(*block);

    Rebuilder rebuild(graph_);
    std::vector<Bit> roots;
    for (const auto& output : outputs) {
        rebuild.context = output.name;
        roots.push_back(rebuild.visit(output.var->wires[output.bit]));
        rebuild.drain();
    }
    BitGraph& g = rebuild.target;

    std::vector<std::string> net_name(g.size());
    for (std::size_t i = 0; i < g.size(); ++i) {
        if (g.node(static_cast<Bit>(i)).op == Op::Input) {
            net_name[i] = g.node(static_cast<Bit>(i)).name;
        }
    }
    std::vector<char> claimed(g.size(), 0);
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        const auto root = static_cast<std::size_t>(roots[i]);
        const Op op = g.node(roots[i]).op;
        if (op != Op::Input && op != Op::Const && net_name[root].empty()) {
            net_name[root] = outputs[i].name;
            claimed[root] = 1;
        }
    }
    std::unordered_set<std::string> used_names;
    for (const auto& net : nets) used_names.insert(net.get_name());
    for (std::size_t i = 3; i < g.size(); ++i) {
        if (!net_name[i].empty()) continue;
        const Node& n = g.node(static_cast<Bit>(i));
        const bool is_state = n.op == Op::Flop || n.op == Op::LatchCell;
        std::string name = is_state && !n.name.empty() ? n.name : "";
        if (name.empty() || !used_names.insert(name).second) {
            name = (is_state ? "_q" : "_n") + std::to_string(i);
        }
        net_name[i] = std::move(name);
    }

    std::vector<Instance> instances;
    bool tie[2] = {false, false};
    const auto net_of = [&](Bit id) -> std::string {
        if (id == kZero || id == kOne) {
            tie[id] = true;
            return id == kOne ? "__tie1" : "__tie0";
        }
        return net_name[static_cast<std::size_t>(id)];
    };
    std::map<Bit, std::string> inverted_clocks;
    int counter = 0;
    const auto make_instance = [&](const std::string& type,
                                   std::vector<PinConnection> pins) {
        instances.emplace_back("__g" + std::to_string(counter++), type, std::move(pins));
    };
    for (std::size_t i = 3; i < g.size(); ++i) {
        const Node& n = g.node(static_cast<Bit>(i));
        const std::string& out = net_name[i];
        switch (n.op) {
            case Op::Not: make_instance("INV", {{"A", net_of(n.a)}, {"Y", out}}); break;
            case Op::And:
                make_instance("AND2", {{"A", net_of(n.a)}, {"B", net_of(n.b)}, {"Y", out}});
                break;
            case Op::Or:
                make_instance("OR2", {{"A", net_of(n.a)}, {"B", net_of(n.b)}, {"Y", out}});
                break;
            case Op::Xor:
                make_instance("XOR2", {{"A", net_of(n.a)}, {"B", net_of(n.b)}, {"Y", out}});
                break;
            case Op::Mux:
                make_instance("MUX2", {{"A0", net_of(n.a)}, {"A1", net_of(n.b)},
                                       {"S", net_of(n.c)}, {"Y", out}});
                break;
            case Op::Flop: {
                std::string clock = net_of(n.b);
                if (n.negedge) {
                    auto found = inverted_clocks.find(n.b);
                    if (found == inverted_clocks.end()) {
                        std::string inverted = "_clkn" + std::to_string(n.b);
                        make_instance("INV", {{"A", clock}, {"Y", inverted}});
                        nets.emplace_back(inverted, NetType::Internal);
                        found = inverted_clocks.emplace(n.b, std::move(inverted)).first;
                    }
                    clock = found->second;
                }
                if (n.c >= 0 && n.d >= 0) {
                    make_instance("DFFSR", {{"D", net_of(n.a)}, {"CLK", clock},
                                            {"RESET_B", net_of(n.c)}, {"SET_B", net_of(n.d)},
                                            {"Y", out}});
                } else if (n.c >= 0) {
                    make_instance(n.reset_value ? "DFFS" : "DFFR",
                                  {{"D", net_of(n.a)}, {"CLK", clock},
                                   {n.reset_value ? "SET_B" : "RESET_B", net_of(n.c)},
                                   {"Y", out}});
                } else {
                    make_instance("DFF", {{"D", net_of(n.a)}, {"CLK", clock}, {"Y", out}});
                }
                break;
            }
            case Op::Tri:
                make_instance("TRIBUF", {{"A", net_of(n.a)}, {"TE_B", net_of(n.b)},
                                         {"Y", out}});
                break;
            case Op::LatchCell:
                make_instance("LATCH", {{"D", net_of(n.a)}, {"GATE", net_of(n.b)}, {"Y", out}});
                break;
            default: break;
        }
    }
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        if (!claimed[static_cast<std::size_t>(roots[i])]) {
            make_instance("BUF", {{"A", net_of(roots[i])}, {"Y", outputs[i].name}});
        }
    }
    for (int value = 0; value < 2; ++value) {
        if (!tie[value]) continue;
        const std::string name = value ? "__tie1" : "__tie0";
        make_instance(value ? "TIEHI" : "TIELO", {{"Y", name}});
        nets.emplace_back(name, NetType::Internal);
    }
    for (std::size_t i = 3; i < g.size(); ++i) {
        const Op op = g.node(static_cast<Bit>(i)).op;
        if (op == Op::Input || claimed[i]) continue;
        nets.emplace_back(net_name[i], NetType::Internal);
    }
    return Module(std::string(top.getDefinition().name), std::move(ports), std::move(nets),
                  std::move(instances), {});
}

}  // namespace

Module detail::lower_behavioral_rtl(
    const std::shared_ptr<ParsedVerilogContext>& context, const std::string& top,
    const std::map<std::string, std::string>& parameters) {
    if (!context || !context->compilation) {
        throw std::runtime_error(
            "behavioral RTL has no retained Slang parse context; create it with parse_verilog");
    }
    std::lock_guard lock(context->mutex);
    auto& driver = context->driver;
    struct DriverOptionsRestore {
        slang::driver::Driver& driver;
        std::vector<std::string> top_modules;
        std::vector<std::string> parameter_overrides;
        bool active = false;

        ~DriverOptionsRestore() {
            if (!active) return;
            driver.options.topModules = std::move(top_modules);
            driver.options.paramOverrides = std::move(parameter_overrides);
        }
    } restore_options{driver, driver.options.topModules,
                      driver.options.paramOverrides};
    slang::ast::Compilation* compilation = context->compilation.get();
    const auto has_top = [&](slang::ast::Compilation& candidate) {
        for (const auto* instance : candidate.getRoot().topInstances) {
            if (instance->getDefinition().name == top) return true;
        }
        return false;
    };

    std::unique_ptr<slang::ast::Compilation> specialized_compilation;
    if (!parameters.empty() || !has_top(*compilation)) {
        restore_options.active = true;
        driver.options.topModules = {top};
        driver.options.paramOverrides.clear();
        for (const auto& [name, value] : parameters) {
            driver.options.paramOverrides.push_back(name + "=" + value);
        }
        specialized_compilation = driver.createCompilation();
        driver.reportCompilation(*specialized_compilation, true);
        if (!driver.reportDiagnostics(true)) {
            throw std::runtime_error(
                "slang failed to elaborate the Verilog design:\n" +
                driver.textDiagClient->getString());
        }
        compilation = specialized_compilation.get();
    }

    const slang::ast::InstanceSymbol* selected_top = nullptr;
    for (const auto* candidate : compilation->getRoot().topInstances) {
        if (candidate->getDefinition().name == top) {
            selected_top = candidate;
            break;
        }
    }
    if (!selected_top) {
        throw std::runtime_error("top module '" + top + "' not found in parsed design");
    }

    Synthesizer synthesizer;
    return synthesizer.run(*selected_top, std::string(selected_top->name));
}

}  // namespace labneura::eda
