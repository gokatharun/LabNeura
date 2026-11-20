#pragma once

#include "labneura/eda/verilog_preprocessor.h"

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace labneura::eda {

struct VerilogSyntaxLocation {
    std::shared_ptr<const VerilogSourceBuffer> source;
    size_t byte_offset = 0;
    VerilogSourceLocation location;
    VerilogPreprocessedOriginKind origin_kind = VerilogPreprocessedOriginKind::Source;
    std::vector<VerilogMacroExpansionFrame> expansion_stack;
};

enum class VerilogSyntaxDiagnosticSeverity : uint8_t { Warning, Error };

struct VerilogSyntaxDiagnostic {
    VerilogSyntaxDiagnosticSeverity severity = VerilogSyntaxDiagnosticSeverity::Error;
    std::string code;
    std::string message;
    VerilogSyntaxLocation location;
};

enum class VerilogExpressionKind : uint8_t {
    Identifier,
    Literal,
    Unary,
    Binary,
    Conditional,
    Call,
    MemberAccess,
    Select,
    Concatenation,
    Replication
};

enum class VerilogSelectKind : uint8_t { Bit, Part, IndexedUp, IndexedDown };

struct VerilogExpression {
    VerilogExpressionKind kind = VerilogExpressionKind::Identifier;
    VerilogSelectKind select_kind = VerilogSelectKind::Bit;
    VerilogSyntaxLocation location;
    std::string text;
    std::unique_ptr<VerilogExpression> first;
    std::unique_ptr<VerilogExpression> second;
    std::unique_ptr<VerilogExpression> third;
    std::vector<std::unique_ptr<VerilogExpression>> operands;
    std::vector<std::optional<std::string>> operand_names;
    std::string member_separator = ".";
};

struct VerilogExpressionRange {
    VerilogSyntaxLocation location;
    std::unique_ptr<VerilogExpression> left;
    std::unique_ptr<VerilogExpression> right;
};

struct VerilogParameterDeclaration {
    VerilogSyntaxLocation location;
    std::string name;
    bool named_override = false;
    bool local_parameter = false;
    std::string type;
    std::vector<VerilogExpressionRange> packed_dimensions;
    std::unique_ptr<VerilogExpression> value;
};

enum class VerilogPortDirection : uint8_t { Unknown, Input, Output, Inout, Ref };

struct VerilogPortDeclaration {
    VerilogSyntaxLocation location;
    VerilogPortDirection direction = VerilogPortDirection::Unknown;
    std::string type;
    std::string name;
    std::vector<VerilogExpressionRange> packed_dimensions;
    std::vector<VerilogExpressionRange> unpacked_dimensions;
};

struct VerilogVariableDeclaration {
    VerilogSyntaxLocation location;
    std::string type;
    std::string name;
    std::vector<VerilogExpressionRange> packed_dimensions;
    std::vector<VerilogExpressionRange> unpacked_dimensions;
    std::unique_ptr<VerilogExpression> initializer;
};

enum class VerilogTypeDefinitionKind : uint8_t { Alias, Enumeration, Structure, Union };

struct VerilogTypeMemberDeclaration {
    VerilogSyntaxLocation location;
    std::string type;
    std::string name;
    std::vector<VerilogExpressionRange> packed_dimensions;
    std::vector<VerilogExpressionRange> unpacked_dimensions;
};

struct VerilogEnumeratorDeclaration {
    VerilogSyntaxLocation location;
    std::string name;
    std::unique_ptr<VerilogExpression> value;
};

struct VerilogTypeDefinition {
    VerilogSyntaxLocation location;
    VerilogTypeDefinitionKind kind = VerilogTypeDefinitionKind::Alias;
    std::string name;
    std::string target_type;
    std::string base_type;
    bool is_packed = false;
    bool is_signed = false;
    std::vector<VerilogExpressionRange> packed_dimensions;
    std::vector<VerilogExpressionRange> unpacked_dimensions;
    std::vector<VerilogTypeMemberDeclaration> members;
    std::vector<VerilogEnumeratorDeclaration> enumerators;
};

struct VerilogContinuousAssignment {
    VerilogSyntaxLocation location;
    std::unique_ptr<VerilogExpression> left;
    std::unique_ptr<VerilogExpression> right;
};

struct VerilogPortConnection {
    VerilogSyntaxLocation location;
    std::optional<std::string> name;
    std::unique_ptr<VerilogExpression> expression;
};

struct VerilogInstance {
    VerilogSyntaxLocation location;
    std::string name;
    std::vector<VerilogExpressionRange> unpacked_dimensions;
    std::vector<VerilogPortConnection> connections;
};

struct VerilogModuleInstantiation {
    VerilogSyntaxLocation location;
    std::string module_type;
    std::vector<VerilogParameterDeclaration> parameter_overrides;
    std::vector<VerilogInstance> instances;
};

enum class VerilogCaseKind : uint8_t { Case, CaseZ, CaseX };
enum class VerilogGenerateKind : uint8_t { Region, If, Case, For };

struct VerilogGenerateConstruct;

struct VerilogGenerateScope {
    VerilogSyntaxLocation location;
    std::string name;
    bool explicitly_named = false;
    std::vector<VerilogParameterDeclaration> parameters;
    std::vector<VerilogTypeDefinition> type_definitions;
    std::vector<VerilogVariableDeclaration> declarations;
    std::vector<VerilogContinuousAssignment> assignments;
    std::vector<VerilogModuleInstantiation> instantiations;
    std::vector<std::unique_ptr<VerilogGenerateConstruct>> constructs;
};

struct VerilogGenerateCaseItem {
    VerilogSyntaxLocation location;
    bool is_default = false;
    std::vector<std::unique_ptr<VerilogExpression>> expressions;
    std::unique_ptr<VerilogGenerateScope> body;
};

struct VerilogGenerateConstruct {
    VerilogGenerateKind kind = VerilogGenerateKind::Region;
    VerilogCaseKind case_kind = VerilogCaseKind::Case;
    VerilogSyntaxLocation location;
    std::unique_ptr<VerilogGenerateScope> region_body;
    std::unique_ptr<VerilogExpression> condition;
    std::unique_ptr<VerilogGenerateScope> true_body;
    std::unique_ptr<VerilogGenerateScope> false_body;
    std::vector<VerilogGenerateCaseItem> case_items;
    std::string loop_variable;
    std::unique_ptr<VerilogExpression> initialization;
    std::unique_ptr<VerilogExpression> loop_condition;
    std::unique_ptr<VerilogExpression> step;
    std::unique_ptr<VerilogGenerateScope> loop_body;
};

enum class VerilogProceduralBlockKind : uint8_t {
    Always,
    AlwaysComb,
    AlwaysFF,
    AlwaysLatch,
    Initial
};

enum class VerilogEventEdge : uint8_t { Any, Posedge, Negedge };

struct VerilogEventTerm {
    VerilogSyntaxLocation location;
    VerilogEventEdge edge = VerilogEventEdge::Any;
    std::unique_ptr<VerilogExpression> expression;
};

struct VerilogEventControl {
    VerilogSyntaxLocation location;
    bool wildcard = false;
    std::vector<VerilogEventTerm> terms;
};

enum class VerilogStatementKind : uint8_t {
    Block,
    If,
    Case,
    Loop,
    ProceduralAssignment,
    TaskCall,
    EventControl,
    DelayControl,
    TimingControl = DelayControl,
    Wait,
    Return,
    Break,
    Continue,
    Empty,
    Unsupported
};

enum class VerilogAssignmentKind : uint8_t { Blocking, Nonblocking };
enum class VerilogLoopKind : uint8_t { For, While, DoWhile, Repeat, Forever };

struct VerilogStatement;

struct VerilogCaseItem {
    VerilogSyntaxLocation location;
    bool is_default = false;
    std::vector<std::unique_ptr<VerilogExpression>> expressions;
    std::unique_ptr<VerilogStatement> statement;
};

struct VerilogStatement {
    VerilogStatementKind kind = VerilogStatementKind::Empty;
    VerilogSyntaxLocation location;
    std::string text;
    std::optional<std::string> label;
    VerilogCaseKind case_kind = VerilogCaseKind::Case;
    VerilogAssignmentKind assignment_kind = VerilogAssignmentKind::Blocking;
    VerilogLoopKind loop_kind = VerilogLoopKind::While;
    std::string loop_variable_name;
    std::string loop_variable_type;
    VerilogSyntaxLocation loop_variable_location;
    std::unique_ptr<VerilogExpression> expression;
    std::unique_ptr<VerilogExpression> call_expression;
    std::unique_ptr<VerilogExpression> left;
    std::unique_ptr<VerilogExpression> right;
    std::unique_ptr<VerilogStatement> loop_initialization;
    std::unique_ptr<VerilogStatement> loop_step;
    std::unique_ptr<VerilogEventControl> event_control;
    std::unique_ptr<VerilogStatement> first_statement;
    std::unique_ptr<VerilogStatement> second_statement;
    std::vector<std::unique_ptr<VerilogStatement>> statements;
    std::vector<VerilogCaseItem> case_items;
};

enum class VerilogCallableKind : uint8_t { Function, Task };
enum class VerilogCallableLifetime : uint8_t { Unspecified, Static, Automatic };

struct VerilogCallableFormalArgument {
    VerilogSyntaxLocation location;
    VerilogPortDirection direction = VerilogPortDirection::Input;
    std::string type;
    std::vector<VerilogExpressionRange> packed_dimensions;
    std::vector<VerilogExpressionRange> unpacked_dimensions;
    std::string name;
};

struct VerilogCallableDeclaration {
    VerilogSyntaxLocation location;
    VerilogCallableKind kind = VerilogCallableKind::Function;
    VerilogCallableLifetime lifetime = VerilogCallableLifetime::Unspecified;
    std::string name;
    std::optional<std::string> return_type;
    std::vector<VerilogExpressionRange> return_packed_dimensions;
    std::vector<VerilogCallableFormalArgument> formal_arguments;
    std::vector<VerilogVariableDeclaration> local_declarations;
    std::unique_ptr<VerilogStatement> body;
};

struct VerilogProceduralBlock {
    VerilogSyntaxLocation location;
    VerilogProceduralBlockKind kind = VerilogProceduralBlockKind::Always;
    std::unique_ptr<VerilogEventControl> event_control;
    std::unique_ptr<VerilogStatement> body;
};

struct VerilogModuleDeclaration {
    VerilogSyntaxLocation location;
    std::string name;
    std::vector<VerilogParameterDeclaration> parameters;
    std::vector<VerilogTypeDefinition> type_definitions;
    std::vector<VerilogPortDeclaration> ports;
    std::vector<VerilogVariableDeclaration> declarations;
    std::vector<VerilogContinuousAssignment> assignments;
    std::vector<VerilogModuleInstantiation> instantiations;
    std::vector<std::unique_ptr<VerilogGenerateConstruct>> generate_constructs;
    std::vector<VerilogProceduralBlock> procedural_blocks;
    std::vector<VerilogCallableDeclaration> callables;
};

struct VerilogInterfaceDeclaration {
    VerilogSyntaxLocation location;
    std::string name;
    std::vector<VerilogParameterDeclaration> parameters;
    std::vector<VerilogTypeDefinition> type_definitions;
    std::vector<VerilogPortDeclaration> ports;
    std::vector<VerilogVariableDeclaration> declarations;
    std::vector<VerilogCallableDeclaration> callables;
};

struct VerilogPackageDeclaration {
    VerilogSyntaxLocation location;
    std::string name;
    std::vector<VerilogParameterDeclaration> parameters;
    std::vector<VerilogTypeDefinition> type_definitions;
    std::vector<VerilogVariableDeclaration> declarations;
    std::vector<VerilogCallableDeclaration> callables;
};

struct VerilogCompilationUnit {
    VerilogSyntaxLocation location;
    std::vector<VerilogModuleDeclaration> modules;
    std::vector<VerilogInterfaceDeclaration> interfaces;
    std::vector<VerilogPackageDeclaration> packages;
};

struct VerilogSyntaxParserOptions {
    VerilogLanguageVersion language_version = VerilogLanguageVersion::SystemVerilog2017;
    VerilogLexerBackend lexer_backend = VerilogLexerBackend::Auto;
};

struct VerilogSyntaxParseResult {
    VerilogCompilationUnit syntax;
    std::vector<VerilogSyntaxDiagnostic> diagnostics;

    bool has_errors() const noexcept;
};

VerilogSyntaxParseResult parse_verilog_syntax(
    const VerilogPreprocessorResult& preprocessed,
    VerilogSyntaxParserOptions options = {});
VerilogSyntaxParseResult parse_verilog_syntax(
    std::string source, VerilogPreprocessorOptions preprocessor_options = {},
    VerilogSyntaxParserOptions parser_options = {}, std::string source_name = {});
VerilogSyntaxParseResult parse_verilog_syntax_file(
    const std::filesystem::path& path,
    VerilogPreprocessorOptions preprocessor_options = {},
    VerilogSyntaxParserOptions parser_options = {});

} // namespace labneura::eda
