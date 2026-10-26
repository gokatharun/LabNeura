#ifndef LABNEURA_EDA_EXPRESSION_H
#define LABNEURA_EDA_EXPRESSION_H

#include <string>
#include <utility>
#include <vector>

namespace labneura::eda {

enum class ExpressionKind {
    Identifier,
    Literal,
    Unary,
    Binary,
    Conditional,
    BitSelect,
    RangeSelect,
    Concatenation,
    Opaque
};

class Expression {
public:
    Expression() = default;
    Expression(ExpressionKind kind, std::string value, std::string source,
               std::vector<Expression> operands = {})
        : kind_(kind),
          value_(std::move(value)),
          source_(std::move(source)),
          operands_(std::move(operands)) {}

    ExpressionKind get_kind() const noexcept { return kind_; }
    const std::string& get_value() const noexcept { return value_; }
    const std::string& get_source() const noexcept { return source_; }
    const std::vector<Expression>& get_operands() const noexcept {
        return operands_;
    }

private:
    ExpressionKind kind_ = ExpressionKind::Opaque;
    std::string value_;
    std::string source_;
    std::vector<Expression> operands_;
};

}  // namespace labneura::eda

#endif
