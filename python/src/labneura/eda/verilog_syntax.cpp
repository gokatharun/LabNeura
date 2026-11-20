#include "labneura/eda/verilog_syntax.h"

#include <algorithm>
#include <utility>

namespace labneura::eda {
namespace {

bool is_identifier(const VerilogToken& token) {
    return token.kind == VerilogTokenKind::Identifier ||
           token.kind == VerilogTokenKind::EscapedIdentifier ||
           token.kind == VerilogTokenKind::SystemIdentifier;
}

int binary_precedence(std::string_view op) {
    if (op == "||")
        return 1;
    if (op == "&&")
        return 2;
    if (op == "|")
        return 3;
    if (op == "^" || op == "~^" || op == "^~")
        return 4;
    if (op == "&")
        return 5;
    if (op == "==" || op == "!=" || op == "===" || op == "!==" || op == "==?" ||
        op == "!=?")
        return 6;
    if (op == "<" || op == "<=" || op == ">" || op == ">=")
        return 7;
    if (op == "<<" || op == ">>" || op == "<<<" || op == ">>>")
        return 8;
    if (op == "+" || op == "-")
        return 9;
    if (op == "*" || op == "/" || op == "%")
        return 10;
    if (op == "**")
        return 11;
    return 0;
}

bool is_type_word(std::string_view word) {
    static constexpr std::string_view words[] = {
        "wire", "tri", "tri0", "tri1", "wand", "wor", "uwire", "reg", "logic", "bit",
        "byte", "shortint", "int", "integer", "longint", "time", "real", "realtime",
        "signed", "unsigned", "var", "const", "static", "automatic"
    };
    return std::find(std::begin(words), std::end(words), word) != std::end(words);
}

bool is_aggregate_type_word(std::string_view word) {
    return word == "enum" || word == "struct" || word == "union";
}

bool is_direction(std::string_view word) {
    return word == "input" || word == "output" || word == "inout" || word == "ref";
}

VerilogPortDirection port_direction(std::string_view word) {
    if (word == "input")
        return VerilogPortDirection::Input;
    if (word == "output")
        return VerilogPortDirection::Output;
    if (word == "inout")
        return VerilogPortDirection::Inout;
    if (word == "ref")
        return VerilogPortDirection::Ref;
    return VerilogPortDirection::Unknown;
}

class Parser {
public:
    Parser(const VerilogPreprocessorResult& preprocessed, VerilogSyntaxParserOptions options) :
        preprocessed_(preprocessed),
        source_(std::make_shared<const VerilogSourceBuffer>(preprocessed.text, "<preprocessed>")),
        lexed_(lex_verilog(source_, {options.language_version, options.lexer_backend})) {
        for (const auto& diagnostic : preprocessed.diagnostics) {
            VerilogSyntaxDiagnostic converted;
            converted.severity = diagnostic.severity == VerilogPreprocessorSeverity::Error
                                     ? VerilogSyntaxDiagnosticSeverity::Error
                                     : VerilogSyntaxDiagnosticSeverity::Warning;
            converted.code = diagnostic.code;
            converted.message = diagnostic.message;
            converted.location = {diagnostic.source, diagnostic.byte_offset, diagnostic.location,
                                  VerilogPreprocessedOriginKind::Source,
                                  diagnostic.expansion_stack};
            result_.diagnostics.push_back(std::move(converted));
        }
        for (const auto& error : lexed_.errors) {
            VerilogSyntaxDiagnostic diagnostic;
            diagnostic.code = "LexicalError";
            diagnostic.message = "invalid token in preprocessed Verilog source";
            diagnostic.location = location(error.byte_offset);
            result_.diagnostics.push_back(std::move(diagnostic));
        }
    }

    VerilogSyntaxParseResult run() {
        result_.syntax.location = peek_location();
        while (!at_end()) {
            if (consume_word("module")) {
                if (auto module = parse_module(previous_location()))
                    result_.syntax.modules.push_back(std::move(*module));
                continue;
            }
            if (consume_word("interface")) {
                if (auto interface = parse_interface(previous_location()))
                    result_.syntax.interfaces.push_back(std::move(*interface));
                continue;
            }
            if (consume_word("package")) {
                if (auto package = parse_package(previous_location()))
                    result_.syntax.packages.push_back(std::move(*package));
                continue;
            }
            error("ExpectedDesignUnit", "expected module, interface, or package declaration",
                  peek_location());
            recover_top_level();
        }
        return std::move(result_);
    }

private:
    VerilogSyntaxLocation location(size_t output_offset) const {
        if (const auto origin = preprocessed_.origin_at(output_offset)) {
            return {origin->source, origin->byte_offset, origin->location, origin->kind,
                    origin->expansion_stack};
        }
        return {source_, output_offset, source_->location(output_offset),
                VerilogPreprocessedOriginKind::Source, {}};
    }

    size_t meaningful_index(size_t from) const {
        while (from < lexed_.tokens.size()) {
            const auto kind = lexed_.tokens[from].kind;
            if (kind != VerilogTokenKind::LineComment &&
                kind != VerilogTokenKind::BlockComment)
                break;
            ++from;
        }
        return from;
    }

    bool looks_like_custom_type_declaration() const {
        size_t lookahead = 0;
        if (!is_identifier(peek()))
            return false;
        ++lookahead;
        if (spelling(peek(lookahead)) == "::") {
            ++lookahead;
            if (!is_identifier(peek(lookahead)))
                return false;
            ++lookahead;
        }
        while (spelling(peek(lookahead)) == "[") {
            size_t depth = 0;
            do {
                const auto token = spelling(peek(lookahead++));
                if (token == "[")
                    ++depth;
                else if (token == "]" && depth)
                    --depth;
            } while (depth && peek(lookahead).kind != VerilogTokenKind::EndOfFile);
        }
        if (!is_identifier(peek(lookahead)))
            return false;
        ++lookahead;
        while (spelling(peek(lookahead)) == "[") {
            size_t depth = 0;
            do {
                const auto token = spelling(peek(lookahead++));
                if (token == "[")
                    ++depth;
                else if (token == "]" && depth)
                    --depth;
            } while (depth && peek(lookahead).kind != VerilogTokenKind::EndOfFile);
        }
        const auto next = spelling(peek(lookahead));
        return next == ";" || next == "=" || next == ",";
    }

    bool looks_like_callable_return_type() const {
        if (is_type_word(spelling(peek())) || is_aggregate_type_word(spelling(peek())))
            return true;
        if (!is_identifier(peek()))
            return false;
        size_t lookahead = 1;
        if (spelling(peek(lookahead)) == "::") {
            ++lookahead;
            if (!is_identifier(peek(lookahead)))
                return false;
            ++lookahead;
        }
        while (spelling(peek(lookahead)) == "[") {
            size_t depth = 0;
            do {
                const auto token = spelling(peek(lookahead++));
                if (token == "[")
                    ++depth;
                else if (token == "]" && depth)
                    --depth;
            } while (depth && peek(lookahead).kind != VerilogTokenKind::EndOfFile);
        }
        return is_identifier(peek(lookahead));
    }

    bool is_data_type_start() const {
        return is_type_word(spelling(peek())) || is_aggregate_type_word(spelling(peek())) ||
               (is_identifier(peek()) &&
                (is_identifier(peek(1)) || spelling(peek(1)) == "::"));
    }

    const VerilogToken& peek(size_t lookahead = 0) const {
        size_t index = cursor_;
        while (lookahead != 0) {
            index = meaningful_index(index);
            if (index >= lexed_.tokens.size() ||
                lexed_.tokens[index].kind == VerilogTokenKind::EndOfFile)
                return lexed_.tokens.back();
            ++index;
            --lookahead;
        }
        index = meaningful_index(index);
        return index < lexed_.tokens.size() ? lexed_.tokens[index] : lexed_.tokens.back();
    }

    std::string_view spelling(const VerilogToken& token) const {
        return lexed_.spelling(token);
    }

    bool at_end() const { return peek().kind == VerilogTokenKind::EndOfFile; }

    bool is(std::string_view value) const { return spelling(peek()) == value; }

    bool consume(std::string_view value) {
        if (!is(value))
            return false;
        advance();
        return true;
    }

    bool consume_word(std::string_view value) {
        if (!is(value))
            return false;
        advance();
        return true;
    }

    void advance() {
        cursor_ = meaningful_index(cursor_);
        if (cursor_ < lexed_.tokens.size() &&
            lexed_.tokens[cursor_].kind != VerilogTokenKind::EndOfFile)
            ++cursor_;
    }

    VerilogSyntaxLocation peek_location() const { return location(peek().byte_offset); }

    VerilogSyntaxLocation previous_location() const {
        size_t index = cursor_;
        if (index == 0)
            return peek_location();
        --index;
        while (index > 0 && (lexed_.tokens[index].kind == VerilogTokenKind::LineComment ||
                             lexed_.tokens[index].kind == VerilogTokenKind::BlockComment))
            --index;
        return location(lexed_.tokens[index].byte_offset);
    }

    void error(std::string code, std::string message, VerilogSyntaxLocation at) {
        result_.diagnostics.push_back(
            {VerilogSyntaxDiagnosticSeverity::Error, std::move(code), std::move(message),
             std::move(at)});
    }

    bool expect(std::string_view value) {
        if (consume(value))
            return true;
        error("ExpectedToken", "expected '" + std::string(value) + "', found '" +
                                   std::string(spelling(peek())) + "'",
              peek_location());
        return false;
    }

    std::optional<std::string> take_name() {
        if (!is_identifier(peek())) {
            error("ExpectedIdentifier", "expected an identifier", peek_location());
            return std::nullopt;
        }
        std::string name(spelling(peek()));
        advance();
        return name;
    }

    void recover_to(std::initializer_list<std::string_view> stops, bool consume_semicolon) {
        while (!at_end()) {
            const auto word = spelling(peek());
            if (std::find(stops.begin(), stops.end(), word) != stops.end())
                return;
            if (is("module") || is("interface") || is("package"))
                return;
            if (word == ";") {
                if (consume_semicolon)
                    advance();
                return;
            }
            advance();
        }
    }

    void recover_top_level() {
        while (!at_end()) {
            if (is("module") || is("interface") || is("package"))
                return;
            advance();
        }
    }

    void recover_design_unit() {
        while (!at_end()) {
            if (is("module") || is("interface") || is("package"))
                return;
            if (is("endmodule") || is("endinterface") || is("endpackage")) {
                advance();
                return;
            }
            advance();
        }
    }

    std::optional<VerilogModuleDeclaration> parse_module(VerilogSyntaxLocation start) {
        VerilogModuleDeclaration module;
        module.location = std::move(start);
        if (is("automatic") || is("static"))
            advance();
        auto name = take_name();
        if (!name) {
            recover_design_unit();
            return std::nullopt;
        }
        module.name = std::move(*name);
        if (consume("#")) {
            if (!expect("(")) {
                recover_design_unit();
                diagnose_undeclared_header_ports(module.ports);
                return module;
            }
            module.parameters = parse_parameter_list(")", false, &module.type_definitions);
        }
        if (consume("("))
            module.ports = parse_port_list(module.type_definitions);
        if (!expect(";")) {
            recover_design_unit();
            diagnose_undeclared_header_ports(module.ports);
            return module;
        }
        while (!at_end() && !is("endmodule")) {
            if (is("module") || is("interface") || is("package")) {
                error("MissingEndmodule", "missing endmodule before next design unit",
                      peek_location());
                diagnose_undeclared_header_ports(module.ports);
                return module;
            }
            const size_t before = cursor_;
            if (is("generate")) {
                module.generate_constructs.push_back(parse_generate_region());
            }
            else if (is("if") || is("case") || is("casez") || is("casex") ||
                     is("for")) {
                module.generate_constructs.push_back(parse_generate_construct());
            }
            else if (is("input") || is("output") || is("inout") || is("ref")) {
                parse_body_port_declarations(module.ports, module.type_definitions, true);
            }
            else if (is("parameter") || is("localparam")) {
                auto declarations = parse_parameter_list(";", false,
                                                        &module.type_definitions);
                module.parameters.insert(module.parameters.end(),
                                         std::make_move_iterator(declarations.begin()),
                                         std::make_move_iterator(declarations.end()));
            }
            else if (is("typedef")) {
                parse_typedef(module.type_definitions);
            }
            else if (is("assign")) {
                parse_assignments(module.assignments);
            }
            else if (is("always") || is("always_comb") || is("always_ff") ||
                     is("always_latch") || is("initial")) {
                if (auto block = parse_procedural_block())
                    module.procedural_blocks.push_back(std::move(*block));
            }
            else if (is("function") || is("task")) {
                module.callables.push_back(
                    parse_callable(module.type_definitions));
            }
            else if (is_type_word(spelling(peek())) ||
                     is_aggregate_type_word(spelling(peek())) ||
                     looks_like_custom_type_declaration()) {
                parse_variable_declaration(module.declarations,
                                           module.type_definitions);
            }
            else if (is_identifier(peek()) || peek().kind == VerilogTokenKind::Keyword) {
                auto instantiation = parse_instantiation();
                if (instantiation)
                    module.instantiations.push_back(std::move(*instantiation));
                else
                    recover_to({"endmodule"}, true);
            }
            else {
                error("UnsupportedModuleItem", "unsupported or malformed module item",
                      peek_location());
                recover_to({"endmodule"}, true);
            }
            if (cursor_ == before) {
                error("ParserStalled", "parser could not recover from module item", peek_location());
                advance();
            }
        }
        if (!expect("endmodule"))
            recover_design_unit();
        if (consume(":"))
            (void)take_name();
        diagnose_undeclared_header_ports(module.ports);
        return module;
    }

    std::string next_generate_block_name() {
        return "genblk" + std::to_string(++generate_block_id_);
    }

    std::unique_ptr<VerilogExpression> make_identifier(std::string name,
                                                       VerilogSyntaxLocation at) {
        auto expression = std::make_unique<VerilogExpression>();
        expression->kind = VerilogExpressionKind::Identifier;
        expression->location = std::move(at);
        expression->text = std::move(name);
        return expression;
    }

    std::unique_ptr<VerilogExpression> make_literal(std::string text,
                                                    VerilogSyntaxLocation at) {
        auto expression = std::make_unique<VerilogExpression>();
        expression->kind = VerilogExpressionKind::Literal;
        expression->location = std::move(at);
        expression->text = std::move(text);
        return expression;
    }

    std::unique_ptr<VerilogExpression> make_binary(
        std::string operation, std::unique_ptr<VerilogExpression> left,
        std::unique_ptr<VerilogExpression> right, VerilogSyntaxLocation at) {
        auto expression = std::make_unique<VerilogExpression>();
        expression->kind = VerilogExpressionKind::Binary;
        expression->location = std::move(at);
        expression->text = std::move(operation);
        expression->first = std::move(left);
        expression->second = std::move(right);
        return expression;
    }

    std::unique_ptr<VerilogExpression> parse_generate_step(
        std::string_view expected_variable) {
        const auto at = peek_location();
        auto target = parse_postfix();
        if (!target || target->kind != VerilogExpressionKind::Identifier) {
            error("InvalidGenerateAssignment", "generate assignment target must be an identifier",
                  at);
            return nullptr;
        }
        if (!expected_variable.empty() && target->text != expected_variable)
            error("InvalidGenerateAssignment",
                  "generate assignment must update loop variable '" +
                      std::string(expected_variable) + "'",
                  target->location);
        const std::string variable = target->text;
        std::unique_ptr<VerilogExpression> value;
        if (consume("=")) {
            value = parse_expression();
        }
        else if (consume("+=") || consume("-=")) {
            const std::string operation(spelling(lexed_.tokens[cursor_ - 1]).substr(0, 1));
            auto rhs = parse_expression();
            value = make_binary(operation, make_identifier(variable, target->location),
                                std::move(rhs), at);
        }
        else if (consume("++") || consume("--")) {
            const std::string operation =
                spelling(lexed_.tokens[cursor_ - 1]) == "++" ? "+" : "-";
            value = make_binary(operation, make_identifier(variable, target->location),
                                make_literal("1", at), at);
        }
        else {
            error("InvalidGenerateAssignment",
                  "generate assignment requires '=', '+=', '-=', '++', or '--'",
                  peek_location());
            return nullptr;
        }
        if (!value) {
            error("MissingGenerateAssignmentValue",
                  "generate assignment requires a constant expression", peek_location());
            return nullptr;
        }
        return make_binary("=", std::move(target), std::move(value), at);
    }

    std::unique_ptr<VerilogGenerateConstruct> parse_generate_region() {
        auto construct = std::make_unique<VerilogGenerateConstruct>();
        construct->kind = VerilogGenerateKind::Region;
        construct->location = peek_location();
        advance();
        construct->region_body = std::make_unique<VerilogGenerateScope>();
        construct->region_body->location = construct->location;
        while (!at_end() && !is("endgenerate")) {
            if (is("endmodule") || is("module") || is("interface") || is("package")) {
                error("ExpectedEndgenerate", "missing 'endgenerate' before design-unit boundary",
                      peek_location());
                return construct;
            }
            const size_t before = cursor_;
            parse_generate_scope_item(*construct->region_body, "endgenerate");
            if (cursor_ == before) {
                error("ParserStalled", "parser could not recover inside generate region",
                      peek_location());
                advance();
            }
        }
        if (!consume("endgenerate"))
            error("ExpectedEndgenerate", "missing 'endgenerate' for generate region",
                  peek_location());
        return construct;
    }

    std::unique_ptr<VerilogGenerateConstruct> parse_generate_construct() {
        auto construct = std::make_unique<VerilogGenerateConstruct>();
        construct->location = peek_location();
        if (consume_word("if")) {
            construct->kind = VerilogGenerateKind::If;
            if (!expect("(")) {
                recover_to({";", "endmodule", "end"}, false);
                return construct;
            }
            if (is(")"))
                error("MissingGenerateCondition", "if-generate requires a condition",
                      peek_location());
            else
                construct->condition = parse_expression();
            if (!expect(")"))
                error("MalformedGenerateCondition", "if-generate condition is missing ')'",
                      peek_location());
            construct->true_body = parse_generate_scope();
            if (consume_word("else"))
                construct->false_body = parse_generate_scope();
            return construct;
        }
        if (consume_word("case")) {
            construct->kind = VerilogGenerateKind::Case;
            construct->case_kind = VerilogCaseKind::Case;
            if (!expect("(")) {
                recover_to({";", "endmodule", "end"}, false);
                return construct;
            }
            construct->condition = parse_expression();
            if (!expect(")"))
                error("MalformedGenerateCase", "case-generate selector is missing ')'",
                      peek_location());
            parse_generate_case_items(*construct);
            return construct;
        }
        if (consume_word("casez") || consume_word("casex")) {
            construct->kind = VerilogGenerateKind::Case;
            construct->case_kind =
                spelling(lexed_.tokens[cursor_ - 1]) == "casez" ? VerilogCaseKind::CaseZ
                                                                 : VerilogCaseKind::CaseX;
            if (!expect("(")) {
                recover_to({";", "endmodule", "end"}, false);
                return construct;
            }
            construct->condition = parse_expression();
            if (!expect(")"))
                error("MalformedGenerateCase", "case-generate selector is missing ')'",
                      peek_location());
            parse_generate_case_items(*construct);
            return construct;
        }
        if (consume_word("for")) {
            construct->kind = VerilogGenerateKind::For;
            if (!expect("(")) {
                recover_to({";", "endmodule", "end"}, false);
                return construct;
            }
            consume_word("genvar");
            const auto variable_location = peek_location();
            auto variable = take_name();
            if (variable)
                construct->loop_variable = std::move(*variable);
            else
                error("MissingGenvar", "for-generate requires a genvar identifier",
                      peek_location());
            if (consume("=")) {
                auto initial_value = parse_expression();
                if (initial_value) {
                    construct->initialization =
                        make_binary("=", make_identifier(construct->loop_variable,
                                                          variable_location),
                                    std::move(initial_value), variable_location);
                }
                else {
                    error("MissingGenerateInitialization",
                          "generate initialization requires a constant expression",
                          peek_location());
                }
            }
            else
                error("MissingGenerateInitialization",
                      "for-generate requires a genvar initialization", peek_location());
            if (!expect(";"))
                error("MalformedGenerateLoop", "expected ';' after generate initialization",
                      peek_location());
            if (is(";"))
                error("MissingGenerateCondition", "for-generate requires a loop condition",
                      peek_location());
            else
                construct->loop_condition = parse_expression();
            if (!expect(";"))
                error("MalformedGenerateLoop", "expected ';' after generate condition",
                      peek_location());
            if (is(")"))
                error("MissingGenerateStep", "for-generate requires a step expression",
                      peek_location());
            else
                construct->step = parse_generate_step(construct->loop_variable);
            if (!expect(")"))
                error("MalformedGenerateLoop", "for-generate header is missing ')'",
                      peek_location());
            construct->loop_body = parse_generate_scope();
            if (construct->loop_body && !construct->loop_body->explicitly_named)
                error("MissingGenerateBlockName",
                      "for-generate loop body must have a named begin block",
                      construct->loop_body->location);
            return construct;
        }
        error("UnsupportedGenerateConstruct", "unsupported generate construct", peek_location());
        recover_to({";", "endgenerate", "endmodule"}, true);
        return construct;
    }

    std::unique_ptr<VerilogGenerateScope> parse_generate_scope() {
        auto scope = std::make_unique<VerilogGenerateScope>();
        scope->location = peek_location();
        if (!consume_word("begin")) {
            scope->name = next_generate_block_name();
            parse_generate_scope_item(*scope, "");
            return scope;
        }
        if (consume(":")) {
            auto name = take_name();
            if (name) {
                scope->name = std::move(*name);
                scope->explicitly_named = true;
            }
            else
                error("MissingGenerateBlockName", "expected a name after 'begin:'",
                      peek_location());
        }
        if (scope->name.empty())
            scope->name = next_generate_block_name();
        while (!at_end() && !is("end")) {
            if (is("endmodule") || is("module") || is("interface") || is("package") ||
                is("endgenerate")) {
                error("ExpectedEnd", "missing 'end' for generate block", peek_location());
                return scope;
            }
            const size_t before = cursor_;
            parse_generate_scope_item(*scope, "end");
            if (cursor_ == before) {
                error("ParserStalled", "parser could not recover inside generate block",
                      peek_location());
                advance();
            }
        }
        if (!consume("end"))
            error("ExpectedEnd", "missing 'end' for generate block", peek_location());
        if (consume(":")) {
            auto closing_name = take_name();
            if (!closing_name || *closing_name != scope->name)
                error("MismatchedGenerateBlockLabel",
                      "closing generate block label does not match opening label",
                      previous_location());
        }
        return scope;
    }

    void parse_generate_scope_item(VerilogGenerateScope& scope,
                                   std::string_view terminator) {
        if (is("generate")) {
            scope.constructs.push_back(parse_generate_region());
        }
        else if (is("if") || is("case") || is("casez") || is("casex") || is("for")) {
            scope.constructs.push_back(parse_generate_construct());
        }
        else if (is("parameter") || is("localparam")) {
            auto parameters = parse_parameter_list(";", false, &scope.type_definitions);
            scope.parameters.insert(scope.parameters.end(),
                                    std::make_move_iterator(parameters.begin()),
                                    std::make_move_iterator(parameters.end()));
        }
        else if (is("assign")) {
            parse_assignments(scope.assignments);
        }
        else if (is("typedef")) {
            parse_typedef(scope.type_definitions);
        }
        else if (is_type_word(spelling(peek())) ||
                 is_aggregate_type_word(spelling(peek())) ||
                 looks_like_custom_type_declaration()) {
            parse_variable_declaration(scope.declarations, scope.type_definitions);
        }
        else if (is_identifier(peek()) || peek().kind == VerilogTokenKind::Keyword) {
            if (auto instantiation = parse_instantiation())
                scope.instantiations.push_back(std::move(*instantiation));
            else
                recover_to({terminator, "endgenerate", "endmodule", "end"}, true);
        }
        else {
            error("UnsupportedGenerateItem", "unsupported or malformed generate item",
                  peek_location());
            recover_to({terminator, "endgenerate", "endmodule", "end"}, true);
        }
    }

    std::optional<VerilogInterfaceDeclaration> parse_interface(VerilogSyntaxLocation start) {
        VerilogInterfaceDeclaration interface;
        interface.location = std::move(start);
        if (is("automatic") || is("static"))
            advance();
        auto name = take_name();
        if (!name) {
            recover_design_unit();
            return std::nullopt;
        }
        interface.name = std::move(*name);
        if (consume("#")) {
            if (expect("("))
                interface.parameters = parse_parameter_list(
                    ")", false, &interface.type_definitions);
        }
        if (consume("("))
            interface.ports = parse_port_list(interface.type_definitions);
        if (!expect(";")) {
            recover_design_unit();
            return interface;
        }
        while (!at_end() && !is("endinterface")) {
            if (is("module") || is("interface") || is("package")) {
                error("MissingEndinterface", "missing endinterface before next design unit",
                      peek_location());
                return interface;
            }
            const size_t before = cursor_;
            if (is("input") || is("output") || is("inout") || is("ref"))
                parse_body_port_declarations(interface.ports, interface.type_definitions, false);
            else if (is("typedef"))
                parse_typedef(interface.type_definitions);
            else if (is("function") || is("task"))
                interface.callables.push_back(parse_callable(interface.type_definitions));
            else if (is_data_type_start() || looks_like_custom_type_declaration())
                parse_variable_declaration(interface.declarations,
                                           interface.type_definitions);
            else {
                error("UnsupportedInterfaceItem", "unsupported or malformed interface item",
                      peek_location());
                recover_to({"endinterface"}, true);
            }
            if (cursor_ == before)
                advance();
        }
        if (!expect("endinterface"))
            recover_design_unit();
        if (consume(":"))
            (void)take_name();
        return interface;
    }

    std::optional<VerilogPackageDeclaration> parse_package(VerilogSyntaxLocation start) {
        VerilogPackageDeclaration package;
        package.location = std::move(start);
        if (is("automatic") || is("static"))
            advance();
        auto name = take_name();
        if (!name) {
            recover_design_unit();
            return std::nullopt;
        }
        package.name = std::move(*name);
        if (!expect(";")) {
            recover_design_unit();
            return package;
        }
        while (!at_end() && !is("endpackage")) {
            if (is("module") || is("interface") || is("package")) {
                error("MissingEndpackage", "missing endpackage before next design unit",
                      peek_location());
                return package;
            }
            const size_t before = cursor_;
            if (is("parameter") || is("localparam")) {
                auto declarations = parse_parameter_list(";", false,
                                                        &package.type_definitions);
                package.parameters.insert(package.parameters.end(),
                                          std::make_move_iterator(declarations.begin()),
                                          std::make_move_iterator(declarations.end()));
            }
            else if (is("typedef"))
                parse_typedef(package.type_definitions);
            else if (is("function") || is("task"))
                package.callables.push_back(parse_callable(package.type_definitions));
            else if (is_data_type_start() || looks_like_custom_type_declaration())
                parse_variable_declaration(package.declarations,
                                           package.type_definitions);
            else {
                error("UnsupportedPackageItem", "unsupported or malformed package item",
                      peek_location());
                recover_to({"endpackage"}, true);
            }
            if (cursor_ == before)
                advance();
        }
        if (!expect("endpackage"))
            recover_design_unit();
        if (consume(":"))
            (void)take_name();
        return package;
    }

    std::vector<VerilogParameterDeclaration> parse_parameter_list(
        std::string_view terminator, bool overrides = false,
        std::vector<VerilogTypeDefinition>* type_definitions = nullptr) {
        std::vector<VerilogParameterDeclaration> parameters;
        std::string inherited_type;
        std::vector<VerilogExpressionRange> inherited_dimensions;
        bool need_parameter = true;
        while (!at_end() && !is(terminator)) {
            const size_t before = cursor_;
            VerilogParameterDeclaration parameter;
            parameter.location = peek_location();
            if (overrides && consume(".")) {
                parameter.named_override = true;
                auto name = take_name();
                if (!name) {
                    recover_to({terminator}, false);
                    break;
                }
                parameter.name = std::move(*name);
                if (!expect("("))
                    break;
                parameter.value = parse_expression();
                expect(")");
                parameters.push_back(std::move(parameter));
                need_parameter = false;
                if (!consume(","))
                    break;
                need_parameter = true;
                continue;
            }
            if (overrides && !is("parameter") && !is("localparam")) {
                parameter.value = parse_expression();
                if (!parameter.value) {
                    recover_to({terminator}, false);
                    break;
                }
                parameters.push_back(std::move(parameter));
                need_parameter = false;
                if (!consume(","))
                    break;
                need_parameter = true;
                continue;
            }
            if (consume_word("localparam"))
                parameter.local_parameter = true;
            else
                consume_word("parameter");
            if (is_data_type_start() && type_definitions) {
                parameter.type = parse_data_type(*type_definitions,
                                                 inherited_dimensions);
                inherited_type = parameter.type;
            }
            else
                parameter.type = inherited_type;
            if (is("["))
                inherited_dimensions = parse_dimensions();
            parameter.packed_dimensions = clone_dimensions(inherited_dimensions);
            auto name = take_name();
            if (!name) {
                recover_to({terminator}, false);
                break;
            }
            parameter.name = std::move(*name);
            if (consume("="))
                parameter.value = parse_expression();
            else
                error("ExpectedParameterValue", "parameter declaration requires an initializer",
                      parameter.location);
            parameters.push_back(std::move(parameter));
            need_parameter = false;
            if (consume(","))
                need_parameter = true;
            else
                break;
            if (cursor_ == before)
                advance();
        }
        if (need_parameter && !parameters.empty())
            error("ExpectedParameter", "expected a parameter after ','", peek_location());
        expect(terminator);
        return parameters;
    }

    std::vector<VerilogPortDeclaration> parse_port_list(
        std::vector<VerilogTypeDefinition>& type_definitions) {
        std::vector<VerilogPortDeclaration> ports;
        VerilogPortDirection inherited_direction = VerilogPortDirection::Unknown;
        std::string inherited_type;
        std::vector<VerilogExpressionRange> inherited_dimensions;
        bool first = true;
        while (!at_end() && !is(")")) {
            const size_t before = cursor_;
            if (!first && !consume(",")) {
                error("ExpectedPortSeparator", "expected ',' between ports", peek_location());
                recover_to({")"}, false);
                break;
            }
            if (!first && is(")")) {
                error("ExpectedPort", "expected a port after ','", peek_location());
                break;
            }
            first = false;
            VerilogPortDeclaration port;
            port.location = peek_location();
            if (is_direction(spelling(peek()))) {
                inherited_direction = port_direction(spelling(peek()));
                advance();
                inherited_type.clear();
                inherited_dimensions.clear();
            }
            port.direction = inherited_direction;
            if (is_data_type_start())
                inherited_type = parse_data_type(type_definitions, inherited_dimensions);
            else if (is("["))
                inherited_dimensions = parse_dimensions();
            port.type = inherited_type;
            port.packed_dimensions = clone_dimensions(inherited_dimensions);
            auto name = take_name();
            if (!name) {
                recover_to({",", ")"}, false);
                if (consume(","))
                    continue;
                break;
            }
            port.name = std::move(*name);
            port.unpacked_dimensions = parse_dimensions();
            if (std::any_of(ports.begin(), ports.end(), [&](const auto& existing) {
                    return existing.name == port.name;
                })) {
                error("DuplicateHeaderPort", "module header contains a duplicate port name",
                      port.location);
            }
            ports.push_back(std::move(port));
            if (cursor_ == before) {
                advance();
                break;
            }
        }
        expect(")");
        return ports;
    }

    static std::vector<VerilogExpressionRange> clone_dimensions(
        const std::vector<VerilogExpressionRange>& dimensions) {
        std::vector<VerilogExpressionRange> result;
        result.reserve(dimensions.size());
        for (const auto& dimension : dimensions) {
            VerilogExpressionRange copy;
            copy.location = dimension.location;
            copy.left = clone_expression(dimension.left.get());
            copy.right = clone_expression(dimension.right.get());
            result.push_back(std::move(copy));
        }
        return result;
    }

    static std::unique_ptr<VerilogExpression> clone_expression(const VerilogExpression* expression) {
        if (!expression)
            return nullptr;
        auto result = std::make_unique<VerilogExpression>();
        result->kind = expression->kind;
        result->select_kind = expression->select_kind;
        result->location = expression->location;
        result->text = expression->text;
        result->member_separator = expression->member_separator;
        result->first = clone_expression(expression->first.get());
        result->second = clone_expression(expression->second.get());
        result->third = clone_expression(expression->third.get());
        for (const auto& operand : expression->operands)
            result->operands.push_back(clone_expression(operand.get()));
        result->operand_names = expression->operand_names;
        return result;
    }

    static bool equivalent_expression(const VerilogExpression* left,
                                     const VerilogExpression* right) {
        if (!left || !right)
            return left == right;
        if (left->kind != right->kind || left->text != right->text ||
            left->member_separator != right->member_separator ||
            left->operands.size() != right->operands.size() ||
            left->operand_names != right->operand_names ||
            !equivalent_expression(left->first.get(), right->first.get()) ||
            !equivalent_expression(left->second.get(), right->second.get()) ||
            !equivalent_expression(left->third.get(), right->third.get()))
            return false;
        for (size_t i = 0; i < left->operands.size(); ++i) {
            if (!equivalent_expression(left->operands[i].get(), right->operands[i].get()))
                return false;
        }
        return true;
    }

    static bool equivalent_dimensions(const std::vector<VerilogExpressionRange>& left,
                                      const std::vector<VerilogExpressionRange>& right) {
        if (left.size() != right.size())
            return false;
        for (size_t i = 0; i < left.size(); ++i) {
            if (!equivalent_expression(left[i].left.get(), right[i].left.get()) ||
                !equivalent_expression(left[i].right.get(), right[i].right.get()))
                return false;
        }
        return true;
    }

    void diagnose_undeclared_header_ports(
        const std::vector<VerilogPortDeclaration>& ports) {
        for (const auto& port : ports) {
            if (port.direction == VerilogPortDirection::Unknown) {
                error("UndeclaredHeaderPort",
                      "module header port '" + port.name +
                          "' has no body port declaration",
                      port.location);
            }
        }
    }

    std::string anonymous_type_name() {
        return "$native_type_" + std::to_string(next_anonymous_type_id_++);
    }

    std::string parse_data_type(std::vector<VerilogTypeDefinition>& definitions,
                                std::vector<VerilogExpressionRange>& packed_dimensions) {
        const auto start = peek_location();
        std::string type;
        if (is("enum"))
            type = parse_enum_type(definitions, start);
        else if (is("struct") || is("union"))
            type = parse_aggregate_type(definitions, start);
        else if (is_type_word(spelling(peek()))) {
            while (is_type_word(spelling(peek()))) {
                if (!type.empty())
                    type.push_back(' ');
                type.append(spelling(peek()));
                advance();
            }
        }
        else if (is_identifier(peek())) {
            type = std::string(spelling(peek()));
            advance();
            if (consume("::")) {
                auto member = take_name();
                if (member)
                    type += "::" + *member;
            }
        }
        packed_dimensions = parse_dimensions();
        return type;
    }

    std::string parse_enum_type(std::vector<VerilogTypeDefinition>& definitions,
                                VerilogSyntaxLocation start) {
        advance();
        VerilogTypeDefinition definition;
        definition.location = std::move(start);
        definition.kind = VerilogTypeDefinitionKind::Enumeration;
        definition.name = anonymous_type_name();
        if (!is("{")) {
            std::vector<VerilogExpressionRange> base_dimensions;
            definition.base_type = parse_data_type(definitions, base_dimensions);
            definition.packed_dimensions = std::move(base_dimensions);
            if (definition.base_type.empty())
                error("InvalidEnumBaseType", "enum base type could not be parsed",
                      definition.location);
        }
        else {
            definition.base_type = "int";
        }
        if (!expect("{")) {
            recover_to({";", "}"}, false);
        }
        while (!at_end() && !is("}")) {
            VerilogEnumeratorDeclaration enumerator;
            enumerator.location = peek_location();
            auto name = take_name();
            if (!name) {
                recover_to({",", "}"}, false);
                if (consume(","))
                    continue;
                break;
            }
            enumerator.name = std::move(*name);
            if (consume("="))
                enumerator.value = parse_expression();
            definition.enumerators.push_back(std::move(enumerator));
            if (!consume(","))
                break;
        }
        if (definition.enumerators.empty())
            error("EmptyEnum", "enum declaration requires at least one enumerator",
                  definition.location);
        expect("}");
        const auto name = definition.name;
        definitions.push_back(std::move(definition));
        return name;
    }

    std::string parse_aggregate_type(std::vector<VerilogTypeDefinition>& definitions,
                                     VerilogSyntaxLocation start) {
        const bool is_union = is("union");
        advance();
        VerilogTypeDefinition definition;
        definition.location = std::move(start);
        definition.kind = is_union ? VerilogTypeDefinitionKind::Union
                                   : VerilogTypeDefinitionKind::Structure;
        definition.name = anonymous_type_name();
        definition.is_packed = consume_word("packed");
        if (consume_word("signed"))
            definition.is_signed = true;
        else
            (void)consume_word("unsigned");
        if (consume_word("tagged"))
            error("UnsupportedTaggedUnion", "tagged unions are not supported",
                  previous_location());
        if (!expect("{"))
            recover_to({"}", ";"}, false);
        while (!at_end() && !is("}")) {
            const size_t before = cursor_;
            VerilogTypeMemberDeclaration member;
            member.location = peek_location();
            member.type = parse_data_type(definitions, member.packed_dimensions);
            if (member.type.empty()) {
                error("InvalidAggregateMember", "aggregate member type could not be parsed",
                      member.location);
                recover_to({";", "}"}, false);
                consume(";");
                continue;
            }
            bool more_members = true;
            while (more_members && !at_end()) {
                VerilogTypeMemberDeclaration named_member;
                named_member.location = member.location;
                named_member.type = member.type;
                named_member.packed_dimensions = clone_dimensions(member.packed_dimensions);
                auto name = take_name();
                if (!name) {
                    recover_to({";", "}"}, false);
                    break;
                }
                named_member.name = std::move(*name);
                named_member.unpacked_dimensions = parse_dimensions();
                definition.members.push_back(std::move(named_member));
                more_members = consume(",");
            }
            expect(";");
            if (cursor_ == before)
                advance();
        }
        if (definition.members.empty())
            error("EmptyAggregate", "struct or union requires at least one member",
                  definition.location);
        expect("}");
        const auto name = definition.name;
        definitions.push_back(std::move(definition));
        return name;
    }

    void parse_typedef(std::vector<VerilogTypeDefinition>& definitions) {
        const auto start = peek_location();
        advance();
        VerilogTypeDefinition definition;
        definition.location = start;
        definition.kind = VerilogTypeDefinitionKind::Alias;
        definition.target_type = parse_data_type(definitions, definition.packed_dimensions);
        auto name = take_name();
        if (!name) {
            recover_to({";"}, true);
            return;
        }
        definition.name = std::move(*name);
        definition.unpacked_dimensions = parse_dimensions();
        expect(";");
        definitions.push_back(std::move(definition));
    }

    void parse_body_port_declarations(
        std::vector<VerilogPortDeclaration>& ports,
        std::vector<VerilogTypeDefinition>& type_definitions,
        bool require_header_port) {
        while (is_direction(spelling(peek()))) {
            const VerilogPortDirection direction = port_direction(spelling(peek()));
            const auto start = peek_location();
            advance();
            std::string type;
            std::vector<VerilogExpressionRange> packed;
            if (is_data_type_start())
                type = parse_data_type(type_definitions, packed);
            else
                packed = parse_dimensions();
            bool more = true;
            while (more && !at_end()) {
                VerilogPortDeclaration port;
                port.location = start;
                port.direction = direction;
                port.type = type;
                port.packed_dimensions = clone_dimensions(packed);
                auto name = take_name();
                if (!name) {
                    recover_to({";"}, true);
                    return;
                }
                port.name = *name;
                port.unpacked_dimensions = parse_dimensions();
                auto existing = std::find_if(ports.begin(), ports.end(), [&](const auto& item) {
                    return item.name == port.name;
                });
                if (existing != ports.end()) {
                    if (existing->direction != VerilogPortDirection::Unknown) {
                        if (!require_header_port) {
                            existing->direction = port.direction;
                            existing->type = port.type;
                            existing->packed_dimensions = std::move(port.packed_dimensions);
                            existing->unpacked_dimensions =
                                std::move(port.unpacked_dimensions);
                        }
                        else {
                            error("DuplicatePortDeclaration",
                                  "port '" + port.name + "' is declared more than once",
                                  port.location);
                            if (existing->direction != port.direction ||
                                existing->type != port.type ||
                                !equivalent_dimensions(existing->packed_dimensions,
                                                       port.packed_dimensions) ||
                                !equivalent_dimensions(existing->unpacked_dimensions,
                                                       port.unpacked_dimensions)) {
                                error("IncompatiblePortDeclaration",
                                      "repeated declaration of port '" + port.name +
                                          "' conflicts with its earlier declaration",
                                      port.location);
                            }
                        }
                    }
                    else {
                        existing->direction = port.direction;
                        existing->type = port.type;
                        existing->packed_dimensions = std::move(port.packed_dimensions);
                        existing->unpacked_dimensions = std::move(port.unpacked_dimensions);
                    }
                }
                else {
                    if (require_header_port) {
                        error("PortDeclarationNotInHeader",
                              "body declaration for '" + port.name +
                                  "' does not match a module header port",
                              port.location);
                    }
                    ports.push_back(std::move(port));
                }
                more = consume(",");
            }
            expect(";");
        }
    }

    std::vector<VerilogExpressionRange> parse_dimensions() {
        std::vector<VerilogExpressionRange> dimensions;
        while (consume("[")) {
            VerilogExpressionRange dimension;
            dimension.location = previous_location();
            dimension.left = parse_expression();
            if (consume(":")) {
                dimension.right = parse_expression();
            }
            expect("]");
            dimensions.push_back(std::move(dimension));
        }
        return dimensions;
    }

    void parse_variable_declaration(
        std::vector<VerilogVariableDeclaration>& declarations,
        std::vector<VerilogTypeDefinition>& type_definitions) {
        const auto start = peek_location();
        std::string type;
        std::vector<VerilogExpressionRange> packed;
        type = parse_data_type(type_definitions, packed);
        bool more = true;
        while (more && !at_end()) {
            VerilogVariableDeclaration declaration;
            declaration.location = start;
            declaration.type = type;
            declaration.packed_dimensions = clone_dimensions(packed);
            auto name = take_name();
            if (!name) {
                recover_to({";"}, true);
                return;
            }
            declaration.name = std::move(*name);
            declaration.unpacked_dimensions = parse_dimensions();
            if (consume("="))
                declaration.initializer = parse_expression();
            declarations.push_back(std::move(declaration));
            more = consume(",");
        }
        expect(";");
    }

    void recover_callable_delimiter(std::string_view delimiter) {
        size_t parentheses = 0;
        size_t brackets = 0;
        size_t braces = 0;
        while (!at_end()) {
            if (parentheses == 0 && brackets == 0 && braces == 0 && is(delimiter)) {
                advance();
                return;
            }
            const auto token = spelling(peek());
            if (token == "(")
                ++parentheses;
            else if (token == ")" && parentheses != 0)
                --parentheses;
            else if (token == "[")
                ++brackets;
            else if (token == "]" && brackets != 0)
                --brackets;
            else if (token == "{")
                ++braces;
            else if (token == "}" && braces != 0)
                --braces;
            advance();
        }
    }

    void recover_callable(bool) {
        while (!at_end()) {
            if (is("endfunction") || is("endtask")) {
                advance();
                if (consume(":"))
                    (void)take_name();
                return;
            }
            if (is("module") || is("interface") || is("package") || is("endmodule") ||
                is("endinterface") || is("endpackage"))
                return;
            advance();
        }
    }

    void parse_callable_formals(VerilogCallableDeclaration& callable,
                                std::vector<VerilogTypeDefinition>& type_definitions) {
        if (consume(")"))
            return;

        VerilogPortDirection inherited_direction = VerilogPortDirection::Unknown;
        std::string inherited_type;
        std::vector<VerilogExpressionRange> inherited_packed_dimensions;
        while (!at_end() && !is(")")) {
            const size_t before = cursor_;
            VerilogCallableFormalArgument argument;
            argument.location = peek_location();
            if (is_direction(spelling(peek()))) {
                inherited_direction = port_direction(spelling(peek()));
                advance();
                inherited_type.clear();
                inherited_packed_dimensions.clear();
            }
            else if (inherited_direction == VerilogPortDirection::Unknown) {
                error("UnsupportedNonAnsiCallableArguments",
                      "callable formals must use ANSI-style direction declarations",
                      peek_location());
                recover_callable_delimiter(")");
                return;
            }
            argument.direction = inherited_direction;
            if (is_data_type_start()) {
                inherited_type =
                    parse_data_type(type_definitions, inherited_packed_dimensions);
            }
            else if (is("["))
                inherited_packed_dimensions = parse_dimensions();

            argument.type = inherited_type.empty() ? "logic" : inherited_type;
            argument.packed_dimensions = clone_dimensions(inherited_packed_dimensions);
            auto name = take_name();
            if (!name) {
                recover_callable_delimiter(")");
                return;
            }
            argument.name = std::move(*name);
            argument.unpacked_dimensions = parse_dimensions();
            if (consume("=")) {
                error("UnsupportedCallableArgumentDefault",
                      "default values on callable formal arguments are not supported",
                      previous_location());
                recover_callable_delimiter(")");
                return;
            }
            callable.formal_arguments.push_back(std::move(argument));
            if (!consume(","))
                break;
            if (is(")")) {
                error("ExpectedCallableFormal",
                      "expected a formal argument after ','", peek_location());
                break;
            }
            if (cursor_ == before)
                advance();
        }
        expect(")");
    }

    std::unique_ptr<VerilogStatement> parse_unsupported_nested_callable_statement() {
        auto statement = std::make_unique<VerilogStatement>();
        statement->kind = VerilogStatementKind::Unsupported;
        statement->location = peek_location();
        const bool function = is("function");
        statement->text = function ? "function" : "task";
        error("UnsupportedNestedCallable",
              "nested function and task declarations are not supported",
              statement->location);
        advance();
        const std::string_view terminator = function ? "endfunction" : "endtask";
        while (!at_end()) {
            if (is(terminator)) {
                advance();
                if (consume(":"))
                    (void)take_name();
                break;
            }
            if (is("endmodule") || is("endinterface") || is("endpackage") ||
                is("endfunction") || is("endtask"))
                break;
            advance();
        }
        return statement;
    }

    VerilogCallableDeclaration parse_callable(
        std::vector<VerilogTypeDefinition>& type_definitions) {
        VerilogCallableDeclaration callable;
        callable.location = peek_location();
        callable.kind = is("function") ? VerilogCallableKind::Function
                                        : VerilogCallableKind::Task;
        const bool function = callable.kind == VerilogCallableKind::Function;
        advance();
        if (consume_word("automatic"))
            callable.lifetime = VerilogCallableLifetime::Automatic;
        else if (consume_word("static"))
            callable.lifetime = VerilogCallableLifetime::Static;

        std::vector<VerilogExpressionRange> return_dimensions;
        if (function) {
            if (looks_like_callable_return_type())
                callable.return_type = parse_data_type(type_definitions, return_dimensions);
            else
                callable.return_type = "logic";
            callable.return_packed_dimensions = std::move(return_dimensions);
        }
        auto name = take_name();
        if (!name) {
            recover_callable(function);
            return callable;
        }
        callable.name = std::move(*name);

        if (!consume("(")) {
            error("UnsupportedNonAnsiCallableArguments",
                  "callable declarations in the supported subset require ANSI-style formal lists",
                  peek_location());
            recover_callable(function);
            return callable;
        }
        parse_callable_formals(callable, type_definitions);
        if (!expect(";")) {
            recover_callable(function);
            return callable;
        }

        auto body = std::make_unique<VerilogStatement>();
        body->kind = VerilogStatementKind::Block;
        body->location = peek_location();

        // The supported subset permits declarations only at the beginning of a
        // callable body. Consume those first, then parse the procedural body.
        while (!at_end() && is_data_type_start()) {
            const size_t before = cursor_;
            parse_variable_declaration(callable.local_declarations, type_definitions);
            if (cursor_ == before) {
                advance();
                break;
            }
        }

        const std::string_view terminator = function ? "endfunction" : "endtask";
        bool mismatched_terminator = false;
        while (!at_end() && !is(terminator)) {
            if (is("endfunction") || is("endtask")) {
                error("MismatchedCallableTerminator",
                      "function/task declaration is closed by the wrong terminator",
                      peek_location());
                advance();
                if (consume(":"))
                    (void)take_name();
                mismatched_terminator = true;
                break;
            }
            if (is("module") || is("interface") || is("package") ||
                is("endmodule") || is("endinterface") || is("endpackage")) {
                error(function ? "MissingEndfunction" : "MissingEndtask",
                      function ? "missing endfunction before design-unit boundary"
                               : "missing endtask before design-unit boundary",
                      peek_location());
                break;
            }
            const size_t before = cursor_;
            if (is("function") || is("task")) {
                body->statements.push_back(parse_unsupported_nested_callable_statement());
            }
            else if (is_direction(spelling(peek()))) {
                error("UnsupportedNonAnsiCallableArguments",
                      "old-style non-ANSI formal declarations are not supported",
                      peek_location());
                recover_callable(function);
                break;
            }
            else if (is_data_type_start()) {
                error("CallableDeclarationAfterStatement",
                      "callable local declarations must precede all statements",
                      peek_location());
                parse_variable_declaration(callable.local_declarations, type_definitions);
            }
            else {
                auto statement = parse_statement();
                if (statement)
                    body->statements.push_back(std::move(statement));
            }
            if (cursor_ == before) {
                error("ParserStalled", "parser could not recover from callable body",
                      peek_location());
                advance();
            }
        }

        if (consume(terminator)) {
            if (consume(":")) {
                auto closing_name = take_name();
                if (closing_name && *closing_name != callable.name)
                    error("MismatchedCallableName",
                          "closing callable name does not match declaration",
                          previous_location());
            }
        }
        else if (!mismatched_terminator && !at_end() && !is("module") &&
                 !is("interface") && !is("package") &&
                 !is("endmodule") && !is("endinterface") && !is("endpackage")) {
            error(function ? "MissingEndfunction" : "MissingEndtask",
                  function ? "missing endfunction" : "missing endtask",
                  peek_location());
        }
        callable.body = std::move(body);
        return callable;
    }

    void parse_assignments(std::vector<VerilogContinuousAssignment>& assignments) {
        advance();
        while (!at_end()) {
            VerilogContinuousAssignment assignment;
            assignment.location = peek_location();
            assignment.left = parse_expression();
            if (!expect("=")) {
                recover_to({";"}, true);
                return;
            }
            assignment.right = parse_expression();
            assignments.push_back(std::move(assignment));
            if (!consume(","))
                break;
        }
        expect(";");
    }

    std::optional<VerilogProceduralBlock> parse_procedural_block() {
        VerilogProceduralBlock block;
        block.location = peek_location();
        if (consume_word("always"))
            block.kind = VerilogProceduralBlockKind::Always;
        else if (consume_word("always_comb"))
            block.kind = VerilogProceduralBlockKind::AlwaysComb;
        else if (consume_word("always_ff"))
            block.kind = VerilogProceduralBlockKind::AlwaysFF;
        else if (consume_word("always_latch"))
            block.kind = VerilogProceduralBlockKind::AlwaysLatch;
        else if (consume_word("initial"))
            block.kind = VerilogProceduralBlockKind::Initial;
        else
            return std::nullopt;

        if (is("@"))
            block.event_control = parse_event_control();
        block.body = parse_statement();
        if (!block.body)
            error("MissingProceduralBody", "expected a statement after procedural block",
                  peek_location());
        return block;
    }

    std::unique_ptr<VerilogStatement> parse_statement() {
        if (is("end") || is("endcase") || is("endfunction") || is("endtask")) {
            error("UnexpectedStatementEnd", "unmatched procedural block terminator",
                  peek_location());
            return nullptr;
        }
        if (at_end() || is("endmodule") || is("module") || is("interface") || is("package") ||
            is("else")) {
            error("MissingStatement", "expected a procedural statement", peek_location());
            return nullptr;
        }
        if (is("begin"))
            return parse_block_statement();
        if (is("function") || is("task"))
            return parse_unsupported_nested_callable_statement();
        if (is("if"))
            return parse_if_statement();
        if (is("case") || is("casez") || is("casex"))
            return parse_case_statement();
        if (is("for") || is("while") || is("repeat") || is("forever") ||
            is("do"))
            return parse_loop_statement();
        if (is("@") || is("#"))
            return parse_timing_control_statement();
        if (is("wait"))
            return parse_wait_statement();
        if (is("return"))
            return parse_return_statement();
        if (is("break") || is("continue"))
            return parse_loop_control_statement();
        if (is("assign") || is("deassign") || is("force") || is("release") ||
            is("fork") || is("disable") || is("assert") || is("assume") || is("cover") ||
            is("deassert"))
            return parse_unsupported_statement();
        if (consume(";")) {
            auto statement = std::make_unique<VerilogStatement>();
            statement->kind = VerilogStatementKind::Empty;
            statement->location = previous_location();
            return statement;
        }
        return parse_procedural_assignment();
    }

    std::unique_ptr<VerilogStatement> parse_block_statement() {
        auto statement = std::make_unique<VerilogStatement>();
        statement->kind = VerilogStatementKind::Block;
        statement->location = peek_location();
        advance();
        if (consume(":"))
            statement->label = take_name();
        while (!at_end() && !is("end")) {
            if (is("module") || is("interface") || is("package") || is("endmodule") ||
                is("endfunction") || is("endtask")) {
                error("ExpectedEnd", "missing 'end' for procedural begin block",
                      peek_location());
                return statement;
            }
            const size_t before = cursor_;
            auto child = parse_statement();
            if (child)
                statement->statements.push_back(std::move(child));
            if (cursor_ == before) {
                error("ParserStalled", "parser could not recover from procedural statement",
                      peek_location());
                advance();
            }
        }
        if (!consume("end")) {
            error("ExpectedEnd", "missing 'end' for procedural begin block", peek_location());
            return statement;
        }
        if (consume(":")) {
            auto closing_label = take_name();
            if (statement->label && closing_label && *statement->label != *closing_label)
                error("MismatchedBlockLabel", "closing block label does not match opening label",
                      previous_location());
        }
        return statement;
    }

    std::unique_ptr<VerilogStatement> parse_if_statement() {
        auto statement = std::make_unique<VerilogStatement>();
        statement->kind = VerilogStatementKind::If;
        statement->location = peek_location();
        advance();
        if (!expect("(")) {
            recover_statement();
            return statement;
        }
        if (is(")") || is(";"))
            error("MissingIfCondition", "expected an expression in if condition", peek_location());
        else
            statement->expression = parse_expression();
        expect(")");
        statement->first_statement = parse_statement();
        if (consume_word("else"))
            statement->second_statement = parse_statement();
        return statement;
    }

    std::unique_ptr<VerilogStatement> parse_case_statement() {
        auto statement = std::make_unique<VerilogStatement>();
        statement->location = peek_location();
        if (consume_word("case"))
            statement->case_kind = VerilogCaseKind::Case;
        else if (consume_word("casez"))
            statement->case_kind = VerilogCaseKind::CaseZ;
        else {
            consume_word("casex");
            statement->case_kind = VerilogCaseKind::CaseX;
        }
        statement->kind = VerilogStatementKind::Case;
        if (!expect("(")) {
            error("MalformedCaseStatement", "case statement requires a selector in parentheses",
                  peek_location());
            recover_statement();
            return statement;
        }
        if (is(")") || is(";"))
            error("MissingCaseExpression", "expected a case selector expression",
                  peek_location());
        else
            statement->expression = parse_expression();
        if (!expect(")"))
            error("MalformedCaseStatement", "case selector is missing its closing ')'",
                  peek_location());

        while (!at_end() && !is("endcase")) {
            if (is("module") || is("interface") || is("package") || is("endmodule")) {
                error("ExpectedEndcase", "missing 'endcase' for case statement",
                      peek_location());
                return statement;
            }
            const size_t before = cursor_;
            VerilogCaseItem item;
            item.location = peek_location();
            if (consume_word("default")) {
                item.is_default = true;
                if (!consume(":"))
                    error("ExpectedCaseColon", "expected ':' after default case item",
                          peek_location());
            }
            else {
                while (!at_end() && !is(":") && !is("endcase")) {
                    if (is(","))
                        error("MissingCaseItemExpression",
                              "expected an expression before ',' in case item",
                              peek_location());
                    else
                        item.expressions.push_back(parse_expression());
                    if (!consume(","))
                        break;
                }
                if (item.expressions.empty())
                    error("MissingCaseItemExpression", "expected a case item expression",
                          peek_location());
                if (!expect(":"))
                    recover_statement();
            }
            item.statement = parse_statement();
            statement->case_items.push_back(std::move(item));
            if (cursor_ == before) {
                error("ParserStalled", "parser could not recover from case item", peek_location());
                advance();
            }
        }
        if (!consume("endcase"))
            error("ExpectedEndcase", "missing 'endcase' for case statement", peek_location());
        return statement;
    }

    std::unique_ptr<VerilogStatement> make_loop_assignment(
        std::unique_ptr<VerilogExpression> target, std::unique_ptr<VerilogExpression> value,
        VerilogSyntaxLocation at) {
        auto assignment = std::make_unique<VerilogStatement>();
        assignment->kind = VerilogStatementKind::ProceduralAssignment;
        assignment->location = std::move(at);
        assignment->left = std::move(target);
        assignment->right = std::move(value);
        assignment->assignment_kind = VerilogAssignmentKind::Blocking;
        return assignment;
    }

    std::unique_ptr<VerilogStatement> parse_loop_assignment(bool step) {
        const auto at = peek_location();
        auto target = parse_postfix();
        if (!target) {
            error(step ? "MissingForStep" : "MissingForInitializer",
                  step ? "expected a loop update assignment" :
                         "expected a loop initializer assignment",
                  peek_location());
            return nullptr;
        }
        if (!is_assignable_target(*target)) {
            error("InvalidAssignmentTarget", "loop assignment target is not assignable",
                  target->location);
        }

        if (consume("=")) {
            auto value = parse_expression();
            if (!value)
                error(step ? "MissingForStep" : "MissingForInitializer",
                      "loop assignment requires a right-hand expression", peek_location());
            return make_loop_assignment(std::move(target), std::move(value), at);
        }

        std::string operation;
        bool implicit_one = false;
        if (consume("++")) {
            operation = "+";
            implicit_one = true;
        }
        else if (consume("--")) {
            operation = "-";
            implicit_one = true;
        }
        else if (consume("+="))
            operation = "+";
        else if (consume("-="))
            operation = "-";
        else if (consume("*="))
            operation = "*";
        else if (consume("/="))
            operation = "/";
        else if (consume("%="))
            operation = "%";
        else if (consume("&="))
            operation = "&";
        else if (consume("|="))
            operation = "|";
        else if (consume("^="))
            operation = "^";
        else if (consume("<<="))
            operation = "<<";
        else if (consume(">>="))
            operation = ">>";
        else if (consume("<<<="))
            operation = "<<<";
        else if (consume(">>>="))
            operation = ">>>";
        else if (consume("**="))
            operation = "**";
        else {
            error(step ? "InvalidForStep" : "InvalidForInitializer",
                  "loop header requires an assignment operator", peek_location());
            return nullptr;
        }

        std::unique_ptr<VerilogExpression> right;
        if (implicit_one) {
            const auto one_location = previous_location();
            right = make_literal("1", one_location);
        }
        else
            right = parse_expression();
        if (!right) {
            error(step ? "MissingForStep" : "MissingForInitializer",
                  "loop update requires a right-hand expression", peek_location());
            return nullptr;
        }

        auto value = std::make_unique<VerilogExpression>();
        value->kind = VerilogExpressionKind::Binary;
        value->location = at;
        value->text = operation;
        value->first = std::make_unique<VerilogExpression>();
        value->first->kind = VerilogExpressionKind::Identifier;
        value->first->location = target->location;
        if (target->kind == VerilogExpressionKind::Identifier)
            value->first->text = target->text;
        else {
            error("UnsupportedForUpdateTarget",
                  "compound loop updates currently require an identifier target",
                  target->location);
            return nullptr;
        }
        value->second = std::move(right);
        return make_loop_assignment(std::move(target), std::move(value), at);
    }

    std::unique_ptr<VerilogStatement> parse_loop_statement() {
        auto statement = std::make_unique<VerilogStatement>();
        statement->kind = VerilogStatementKind::Loop;
        statement->location = peek_location();

        if (consume_word("do")) {
            statement->loop_kind = VerilogLoopKind::DoWhile;
            statement->first_statement = parse_statement();
            if (!consume_word("while")) {
                error("MalformedDoWhileLoop", "do loop requires a trailing while condition",
                      peek_location());
                recover_statement();
                return statement;
            }
            if (!expect("("))
                error("MalformedDoWhileLoop", "do-while condition requires parentheses",
                      peek_location());
            if (is(")"))
                error("MissingDoWhileCondition", "do-while loop requires a condition",
                      peek_location());
            else
                statement->expression = parse_expression();
            if (!expect(")"))
                error("MalformedDoWhileLoop", "do-while condition is missing ')'",
                      peek_location());
            if (!expect(";"))
                error("MalformedDoWhileLoop", "do-while loop requires a trailing ';'",
                      peek_location());
            return statement;
        }
        if (consume_word("for")) {
            statement->loop_kind = VerilogLoopKind::For;
            if (!expect("(")) {
                recover_statement();
                return statement;
            }
            if (is_type_word(spelling(peek()))) {
                const auto type_start = cursor_;
                while (is_type_word(spelling(peek()))) {
                    if (statement->loop_variable_type.size() != 0)
                        statement->loop_variable_type.push_back(' ');
                    statement->loop_variable_type.append(spelling(peek()));
                    advance();
                }
                if (cursor_ == type_start) {
                    error("InvalidForInitializer", "invalid loop variable declaration",
                          peek_location());
                }
                statement->loop_variable_location = peek_location();
                if (auto name = take_name())
                    statement->loop_variable_name = std::move(*name);
                if (!expect("="))
                    error("InvalidForInitializer",
                          "declared loop variable requires an initializer", peek_location());
                auto initial_value = parse_expression();
                auto target = make_identifier(statement->loop_variable_name,
                                              statement->loop_variable_location);
                statement->loop_initialization =
                    make_loop_assignment(std::move(target), std::move(initial_value),
                                         statement->location);
            }
            else if (!is(";"))
                statement->loop_initialization = parse_loop_assignment(false);

            if (!expect(";")) {
                error("MalformedForLoop", "expected ';' after for-loop initializer",
                      peek_location());
                recover_to({";", ")", "endmodule"}, false);
                consume(";");
            }
            if (!is(";"))
                statement->expression = parse_expression();
            if (!expect(";")) {
                error("MalformedForLoop", "expected ';' after for-loop condition",
                      peek_location());
                recover_to({";", ")", "endmodule"}, false);
                consume(";");
            }
            if (!is(")"))
                statement->loop_step = parse_loop_assignment(true);
            if (!expect(")"))
                error("MalformedForLoop", "for-loop header is missing ')'", peek_location());
        }
        else if (consume_word("while")) {
            statement->loop_kind = VerilogLoopKind::While;
            if (!expect("("))
                error("MalformedWhileLoop", "while loop requires a parenthesized condition",
                      peek_location());
            if (is(")"))
                error("MissingWhileCondition", "while loop requires a condition", peek_location());
            else
                statement->expression = parse_expression();
            if (!expect(")"))
                error("MalformedWhileLoop", "while condition is missing ')'", peek_location());
        }
        else if (consume_word("repeat")) {
            statement->loop_kind = VerilogLoopKind::Repeat;
            if (!expect("("))
                error("MalformedRepeatLoop", "repeat loop requires a parenthesized count",
                      peek_location());
            if (is(")"))
                error("MissingRepeatCount", "repeat loop requires a count", peek_location());
            else
                statement->expression = parse_expression();
            if (!expect(")"))
                error("MalformedRepeatLoop", "repeat count is missing ')'", peek_location());
        }
        else {
            consume_word("forever");
            statement->loop_kind = VerilogLoopKind::Forever;
        }
        statement->first_statement = parse_statement();
        return statement;
    }

    std::unique_ptr<VerilogStatement> parse_loop_control_statement() {
        auto statement = std::make_unique<VerilogStatement>();
        statement->location = peek_location();
        if (consume_word("break"))
            statement->kind = VerilogStatementKind::Break;
        else {
            consume_word("continue");
            statement->kind = VerilogStatementKind::Continue;
        }
        expect(";");
        return statement;
    }

    std::unique_ptr<VerilogEventControl> parse_event_control() {
        auto control = std::make_unique<VerilogEventControl>();
        control->location = peek_location();
        if (!consume("@"))
            return control;
        if (consume("*")) {
            control->wildcard = true;
            return control;
        }
        if (!is("(")) {
            VerilogEventTerm term;
            term.location = peek_location();
            term.expression = parse_postfix();
            if (!term.expression) {
                error("MissingEventExpression",
                      "expected a signal name after '@'", term.location);
                return control;
            }
            control->terms.push_back(std::move(term));
            return control;
        }
        if (!expect("("))
            return control;
        if (consume("*")) {
            control->wildcard = true;
            if (!expect(")"))
                recover_statement();
            return control;
        }
        while (!at_end() && !is(")")) {
            if (is(",") || is("or")) {
                error("MissingEventExpression", "expected an expression in event control",
                      peek_location());
                advance();
                continue;
            }
            VerilogEventTerm term;
            term.location = peek_location();
            if (consume_word("posedge"))
                term.edge = VerilogEventEdge::Posedge;
            else if (consume_word("negedge"))
                term.edge = VerilogEventEdge::Negedge;
            if (is(")") || is(",") || is("or"))
                error("MissingEventExpression", "expected an expression after event edge",
                      peek_location());
            else
                term.expression = parse_expression();
            control->terms.push_back(std::move(term));
            if (consume(",") || consume_word("or"))
                continue;
            if (!is(")")) {
                error("ExpectedEventSeparator", "expected 'or', ',' or ')' in event control",
                      peek_location());
                recover_to({")", ";"}, false);
                break;
            }
        }
        if (!expect(")"))
            error("MalformedEventControl", "event control is missing its closing ')'",
                  peek_location());
        if (control->terms.empty() && !control->wildcard)
            error("MissingEventExpression", "event control requires at least one event term",
                  control->location);
        return control;
    }

    std::unique_ptr<VerilogStatement> parse_timing_control_statement() {
        auto statement = std::make_unique<VerilogStatement>();
        statement->location = peek_location();
        if (is("@")) {
            statement->kind = VerilogStatementKind::EventControl;
            statement->event_control = parse_event_control();
        }
        else {
            statement->kind = VerilogStatementKind::DelayControl;
            advance();
            if (consume("(")) {
                if (is(")"))
                    error("MissingDelayExpression", "expected a delay expression",
                          peek_location());
                else
                    statement->expression = parse_expression();
                if (!expect(")"))
                    error("MalformedDelayControl", "delay control is missing its closing ')'",
                          peek_location());
            }
            else if (!is(";"))
                statement->expression = parse_expression();
            else
                error("MissingDelayExpression", "expected a delay expression",
                      peek_location());
        }
        statement->first_statement = parse_statement();
        return statement;
    }

    std::unique_ptr<VerilogStatement> parse_wait_statement() {
        auto statement = std::make_unique<VerilogStatement>();
        statement->kind = VerilogStatementKind::Wait;
        statement->location = peek_location();
        advance();
        if (!expect("(")) {
            recover_statement();
            return statement;
        }
        if (is(")") || is(";"))
            error("MissingWaitCondition", "expected an expression in wait condition",
                  peek_location());
        else
            statement->expression = parse_expression();
        expect(")");
        statement->first_statement = parse_statement();
        return statement;
    }

    std::unique_ptr<VerilogStatement> parse_return_statement() {
        auto statement = std::make_unique<VerilogStatement>();
        statement->kind = VerilogStatementKind::Return;
        statement->location = peek_location();
        advance();
        if (!is(";") && !at_end())
            statement->expression = parse_expression();
        if (!consume(";")) {
            error("ExpectedSemicolon", "expected ';' after return statement",
                  peek_location());
            recover_statement();
        }
        return statement;
    }

    std::unique_ptr<VerilogStatement> parse_procedural_assignment() {
        auto statement = std::make_unique<VerilogStatement>();
        statement->kind = VerilogStatementKind::ProceduralAssignment;
        statement->location = peek_location();
        statement->left = parse_postfix();
        if (!statement->left) {
            recover_statement();
            return statement;
        }
        if (statement->left->kind == VerilogExpressionKind::Call && consume(";")) {
            statement->kind = VerilogStatementKind::TaskCall;
            statement->call_expression = std::move(statement->left);
            return statement;
        }
        if (!is_assignable_target(*statement->left))
            error("InvalidAssignmentTarget", "procedural assignment target is not assignable",
                  statement->left->location);
        if (consume("="))
            statement->assignment_kind = VerilogAssignmentKind::Blocking;
        else if (consume("<="))
            statement->assignment_kind = VerilogAssignmentKind::Nonblocking;
        else {
            error("ExpectedAssignmentOperator",
                  "expected blocking '=' or nonblocking '<=' in procedural assignment",
                  peek_location());
            recover_statement();
            return statement;
        }
        if (is(";") || is("end") || is("else") || is("endcase") || at_end())
            error("MissingAssignmentExpression", "expected an expression after assignment operator",
                  peek_location());
        else
            statement->right = parse_expression();
        if (!consume(";")) {
            error("ExpectedSemicolon", "expected ';' after procedural assignment",
                  peek_location());
            recover_statement();
        }
        return statement;
    }

    static bool is_assignable_target(const VerilogExpression& expression) {
        switch (expression.kind) {
            case VerilogExpressionKind::Identifier:
                return true;
            case VerilogExpressionKind::MemberAccess:
            case VerilogExpressionKind::Select:
                return expression.first && is_assignable_target(*expression.first);
            case VerilogExpressionKind::Concatenation:
                return !expression.operands.empty() &&
                       std::all_of(expression.operands.begin(), expression.operands.end(),
                                   [](const auto& operand) {
                                       return operand && is_assignable_target(*operand);
                                   });
            default:
                return false;
        }
    }

    std::unique_ptr<VerilogStatement> parse_unsupported_statement() {
        auto statement = std::make_unique<VerilogStatement>();
        statement->kind = VerilogStatementKind::Unsupported;
        statement->location = peek_location();
        statement->text = std::string(spelling(peek()));
        error("UnsupportedProceduralStatement",
              "unsupported procedural statement '" + statement->text + "'", statement->location);
        int parentheses = 0;
        int brackets = 0;
        int braces = 0;
        int blocks = 0;
        int cases = 0;
        while (!at_end()) {
            if (is("module") || is("interface") || is("package") || is("endmodule"))
                break;
            if (is("end") && blocks == 0)
                break;
            if (is("else") && blocks == 0 && cases == 0)
                break;
            if (is("endcase") && cases == 0)
                break;
            const auto token = spelling(peek());
            if (token == "(")
                ++parentheses;
            else if (token == ")" && parentheses > 0)
                --parentheses;
            else if (token == "[")
                ++brackets;
            else if (token == "]" && brackets > 0)
                --brackets;
            else if (token == "{")
                ++braces;
            else if (token == "}" && braces > 0)
                --braces;
            else if (token == "begin")
                ++blocks;
            else if (token == "end" && blocks > 0)
                --blocks;
            else if (token == "case" || token == "casez" || token == "casex")
                ++cases;
            else if (token == "endcase" && cases > 0)
                --cases;
            else if (token == ";" && parentheses == 0 && brackets == 0 && braces == 0 &&
                     blocks == 0 && cases == 0) {
                advance();
                break;
            }
            advance();
            if (blocks == 0 && cases == 0 && parentheses == 0 && brackets == 0 && braces == 0 &&
                (token == "end" || token == "endcase")) {
                if (consume(":"))
                    (void)take_name();
                break;
            }
        }
        return statement;
    }

    void recover_statement() {
        while (!at_end()) {
            if (is("module") || is("interface") || is("package") || is("endmodule") ||
                is("end") || is("else") || is("endcase"))
                return;
            if (consume(";"))
                return;
            advance();
        }
    }

    std::optional<VerilogModuleInstantiation> parse_instantiation() {
        VerilogModuleInstantiation instantiation;
        instantiation.location = peek_location();
        auto type = take_name();
        if (!type)
            return std::nullopt;
        instantiation.module_type = std::move(*type);
        if (consume("#")) {
            if (!expect("("))
                return std::nullopt;
            instantiation.parameter_overrides = parse_parameter_list(")", true);
        }
        bool another = true;
        while (another && !at_end()) {
            VerilogInstance instance;
            instance.location = peek_location();
            auto name = take_name();
            if (!name)
                return std::nullopt;
            instance.name = std::move(*name);
            instance.unpacked_dimensions = parse_dimensions();
            if (!expect("("))
                return std::nullopt;
            instance.connections = parse_connections();
            if (!expect(")"))
                return std::nullopt;
            instantiation.instances.push_back(std::move(instance));
            another = consume(",");
        }
        if (!expect(";"))
            return std::nullopt;
        return instantiation;
    }

    std::vector<VerilogPortConnection> parse_connections() {
        std::vector<VerilogPortConnection> connections;
        bool first = true;
        while (!at_end() && !is(")")) {
            if (!first && !consume(",")) {
                error("ExpectedConnectionSeparator", "expected ',' between connections",
                      peek_location());
                recover_to({")"}, false);
                break;
            }
            if (!first && is(")")) {
                error("ExpectedConnection", "expected a connection after ','", peek_location());
                break;
            }
            first = false;
            VerilogPortConnection connection;
            connection.location = peek_location();
            if (consume(".")) {
                auto name = take_name();
                if (name)
                    connection.name = std::move(*name);
                if (consume("(")) {
                    if (!is(")"))
                        connection.expression = parse_expression();
                    expect(")");
                }
                else {
                    error("ExpectedToken", "expected '(' after named connection",
                          peek_location());
                }
            }
            else if (is(",")) {
                connections.push_back(std::move(connection));
                continue;
            }
            else {
                connection.expression = parse_expression();
            }
            connections.push_back(std::move(connection));
        }
        return connections;
    }

    std::unique_ptr<VerilogExpression> parse_expression(int minimum_precedence = 1) {
        auto left = parse_unary();
        if (!left)
            return nullptr;
        while (!at_end()) {
            if (is("?") && minimum_precedence <= 1) {
                const auto at = peek_location();
                advance();
                auto when_true = parse_expression();
                if (!expect(":"))
                    return left;
                auto when_false = parse_expression(1);
                auto conditional = std::make_unique<VerilogExpression>();
                conditional->kind = VerilogExpressionKind::Conditional;
                conditional->location = at;
                conditional->first = std::move(left);
                conditional->second = std::move(when_true);
                conditional->third = std::move(when_false);
                left = std::move(conditional);
                continue;
            }
            const auto op = spelling(peek());
            const int precedence = binary_precedence(op);
            if (precedence < minimum_precedence)
                break;
            const auto at = peek_location();
            const std::string operation(op);
            advance();
            auto right = parse_expression(precedence + (op == "**" ? 0 : 1));
            if (!right) {
                error("ExpectedExpression", "expected expression after '" + operation + "'",
                      peek_location());
                break;
            }
            auto binary = std::make_unique<VerilogExpression>();
            binary->kind = VerilogExpressionKind::Binary;
            binary->location = at;
            binary->text = operation;
            binary->first = std::move(left);
            binary->second = std::move(right);
            left = std::move(binary);
        }
        return left;
    }

    std::unique_ptr<VerilogExpression> parse_unary() {
        const auto op = spelling(peek());
        if (op == "+" || op == "-" || op == "!" || op == "~" || op == "&" || op == "~&" ||
            op == "|" || op == "~|" || op == "^" || op == "~^" || op == "^~") {
            const auto at = peek_location();
            advance();
            auto operand = parse_unary();
            auto expression = std::make_unique<VerilogExpression>();
            expression->kind = VerilogExpressionKind::Unary;
            expression->location = at;
            expression->text = std::string(op);
            expression->first = std::move(operand);
            return expression;
        }
        return parse_postfix();
    }

    std::unique_ptr<VerilogExpression> parse_postfix() {
        auto expression = parse_primary();
        if (!expression)
            return nullptr;
        while (!at_end()) {
            if (consume("(")) {
                auto call = std::make_unique<VerilogExpression>();
                call->kind = VerilogExpressionKind::Call;
                call->location = expression->location;
                call->first = std::move(expression);
                enum class ArgumentStyle { Unknown, Positional, Named };
                ArgumentStyle style = ArgumentStyle::Unknown;
                bool recovered_call_close = false;
                while (!at_end() && !is(")")) {
                    if (is(".")) {
                        const auto argument_location = peek_location();
                        advance();
                        auto name = take_name();
                        if (!name) {
                            error("MalformedNamedActualArgument",
                                  "expected a formal name after '.'", argument_location);
                            recover_callable_delimiter(")");
                            recovered_call_close = true;
                            break;
                        }
                        if (style == ArgumentStyle::Positional)
                            error("MixedCallArgumentStyles",
                                  "positional and named actual arguments cannot be mixed",
                                  argument_location);
                        style = ArgumentStyle::Named;
                        if (!consume("(")) {
                            error("MalformedNamedActualArgument",
                                  "expected '(' after named actual argument",
                                  peek_location());
                            recover_callable_delimiter(")");
                            recovered_call_close = true;
                            break;
                        }
                        if (is(")")) {
                            error("MissingNamedCallActual",
                                  "named actual argument requires an expression",
                                  argument_location);
                            advance();
                        }
                        else {
                            auto argument = parse_expression();
                            if (!argument) {
                                error("MalformedNamedActualArgument",
                                      "expected an expression in named actual argument",
                                      peek_location());
                                recover_to({")"}, false);
                                if (!consume(")")) {
                                    recovered_call_close = true;
                                    break;
                                }
                            }
                            else if (!is(")")) {
                                error("MalformedNamedActualArgument",
                                      "expected ')' after named actual expression",
                                      peek_location());
                                recover_to({")"}, false);
                                if (!consume(")")) {
                                    recovered_call_close = true;
                                    break;
                                }
                            }
                            else {
                                advance();
                                if (std::find(call->operand_names.begin(),
                                              call->operand_names.end(),
                                              std::optional<std::string>(*name)) !=
                                    call->operand_names.end()) {
                                    error("DuplicateNamedActualArgument",
                                          "named actual argument '" + *name +
                                              "' appears more than once",
                                          argument_location);
                                }
                                call->operands.push_back(std::move(argument));
                                call->operand_names.emplace_back(std::move(*name));
                            }
                        }
                    }
                    else {
                        if (is(",")) {
                            error("ExpectedCallArgument", "expected a call argument",
                                  peek_location());
                            advance();
                            continue;
                        }
                        if (style == ArgumentStyle::Named)
                            error("MixedCallArgumentStyles",
                                  "positional and named actual arguments cannot be mixed",
                                  peek_location());
                        style = ArgumentStyle::Positional;
                        auto argument = parse_expression();
                        if (!argument) {
                            error("ExpectedCallArgument",
                                  "expected an expression for call argument",
                                  peek_location());
                            recover_callable_delimiter(")");
                            recovered_call_close = true;
                            break;
                        }
                        call->operands.push_back(std::move(argument));
                        call->operand_names.emplace_back(std::nullopt);
                    }
                    if (!consume(","))
                        break;
                    if (is(")")) {
                        error("ExpectedCallArgument", "expected an argument after ','",
                              peek_location());
                        break;
                    }
                }
                if (!recovered_call_close)
                    expect(")");
                expression = std::move(call);
            }
            else if (consume("[")) {
                auto select = std::make_unique<VerilogExpression>();
                select->kind = VerilogExpressionKind::Select;
                select->location = expression->location;
                select->first = std::move(expression);
                select->second = parse_expression();
                if (consume(":")) {
                    select->select_kind = VerilogSelectKind::Part;
                    select->third = parse_expression();
                } else if (consume("+:")) {
                    select->select_kind = VerilogSelectKind::IndexedUp;
                    select->third = parse_expression();
                } else if (consume("-:")) {
                    select->select_kind = VerilogSelectKind::IndexedDown;
                    select->third = parse_expression();
                }
                expect("]");
                expression = std::move(select);
            }
            else if (consume(".") || consume("::")) {
                const std::string separator(spelling(lexed_.tokens[cursor_ - 1]));
                auto member = std::make_unique<VerilogExpression>();
                member->kind = VerilogExpressionKind::MemberAccess;
                member->location = expression->location;
                member->first = std::move(expression);
                member->member_separator = separator;
                const auto member_name = take_name();
                if (member_name)
                    member->text = *member_name;
                expression = std::move(member);
            }
            else {
                break;
            }
        }
        return expression;
    }

    std::unique_ptr<VerilogExpression> parse_primary() {
        const auto token = peek();
        const auto text = spelling(token);
        if (token.kind == VerilogTokenKind::DecimalNumber ||
            token.kind == VerilogTokenKind::BasedNumber ||
            token.kind == VerilogTokenKind::UnbasedNumber ||
            token.kind == VerilogTokenKind::RealNumber ||
            token.kind == VerilogTokenKind::StringLiteral) {
            advance();
            auto literal = std::make_unique<VerilogExpression>();
            literal->kind = VerilogExpressionKind::Literal;
            literal->location = location(token.byte_offset);
            literal->text = std::string(text);
            return literal;
        }
        if (is_identifier(token) || token.kind == VerilogTokenKind::Keyword) {
            advance();
            auto identifier = std::make_unique<VerilogExpression>();
            identifier->kind = VerilogExpressionKind::Identifier;
            identifier->location = location(token.byte_offset);
            identifier->text = std::string(text);
            return identifier;
        }
        if (consume("(")) {
            auto expression = parse_expression();
            expect(")");
            return expression;
        }
        if (consume("{")) {
            auto concatenation = std::make_unique<VerilogExpression>();
            concatenation->location = location(token.byte_offset);
            auto first = parse_expression();
            if (consume("{")) {
                concatenation->kind = VerilogExpressionKind::Replication;
                concatenation->first = std::move(first);
                while (!at_end() && !is("}")) {
                    concatenation->operands.push_back(parse_expression());
                    if (!consume(","))
                        break;
                }
                expect("}");
            }
            else {
                concatenation->kind = VerilogExpressionKind::Concatenation;
                if (first)
                    concatenation->operands.push_back(std::move(first));
                while (consume(",")) {
                    if (is("}")) {
                        error("ExpectedConcatenationOperand",
                              "expected expression after ',' in concatenation",
                              peek_location());
                        break;
                    }
                    concatenation->operands.push_back(parse_expression());
                }
            }
            expect("}");
            return concatenation;
        }
        error("ExpectedExpression", "expected expression, found '" + std::string(text) + "'",
              location(token.byte_offset));
        if (!at_end())
            advance();
        return nullptr;
    }

    void parse_generate_case_items(VerilogGenerateConstruct& construct) {
        while (!at_end() && !is("endcase")) {
            if (is("endmodule") || is("module") || is("interface") || is("package")) {
                error("ExpectedEndcase", "missing endcase for case-generate", peek_location());
                return;
            }
            const size_t before = cursor_;
            VerilogGenerateCaseItem item;
            item.location = peek_location();
            if (consume_word("default")) {
                item.is_default = true;
                if (consume(":"))
                    item.body = parse_generate_scope();
                else
                    error("ExpectedCaseColon", "expected ':' after generate default",
                          peek_location());
            }
            else {
                while (!at_end() && !is(":") && !is("endcase")) {
                    item.expressions.push_back(parse_expression());
                    if (!consume(","))
                        break;
                }
                if (!expect(":"))
                    error("ExpectedCaseColon", "expected ':' after generate case item",
                          peek_location());
                item.body = parse_generate_scope();
            }
            construct.case_items.push_back(std::move(item));
            if (cursor_ == before) {
                error("ParserStalled", "parser could not recover from generate case item",
                      peek_location());
                advance();
            }
        }
        if (!consume("endcase"))
            error("ExpectedEndcase", "missing endcase for case-generate", peek_location());
    }

    const VerilogPreprocessorResult& preprocessed_;
    std::shared_ptr<const VerilogSourceBuffer> source_;
    VerilogLexResult lexed_;
    size_t cursor_ = 0;
    size_t generate_block_id_ = 0;
    VerilogSyntaxParseResult result_;
    size_t next_anonymous_type_id_ = 0;
};

} // namespace

bool VerilogSyntaxParseResult::has_errors() const noexcept {
    return std::any_of(diagnostics.begin(), diagnostics.end(), [](const auto& diagnostic) {
        return diagnostic.severity == VerilogSyntaxDiagnosticSeverity::Error;
    });
}

VerilogSyntaxParseResult parse_verilog_syntax(
    const VerilogPreprocessorResult& preprocessed, VerilogSyntaxParserOptions options) {
    return Parser(preprocessed, options).run();
}

VerilogSyntaxParseResult parse_verilog_syntax(
    std::string source, VerilogPreprocessorOptions preprocessor_options,
    VerilogSyntaxParserOptions parser_options, std::string source_name) {
    preprocessor_options.language_version = parser_options.language_version;
    auto preprocessed = preprocess_verilog(std::move(source), std::move(preprocessor_options),
                                            std::move(source_name));
    return parse_verilog_syntax(preprocessed, parser_options);
}

VerilogSyntaxParseResult parse_verilog_syntax_file(
    const std::filesystem::path& path, VerilogPreprocessorOptions preprocessor_options,
    VerilogSyntaxParserOptions parser_options) {
    preprocessor_options.language_version = parser_options.language_version;
    auto preprocessed = preprocess_verilog_file(path, std::move(preprocessor_options));
    return parse_verilog_syntax(preprocessed, parser_options);
}

} // namespace labneura::eda
