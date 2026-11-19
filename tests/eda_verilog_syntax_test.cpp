#include "labneura/eda/verilog_syntax.h"
#include "labneura/eda/verilog_semantic.h"

#include <slang/syntax/SyntaxTree.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

using namespace labneura::eda;

void require(bool condition, std::string_view message) {
    if (!condition)
        throw std::runtime_error(std::string(message));
}

bool has_error_code(const VerilogSyntaxParseResult& result, std::string_view code) {
    return std::any_of(result.diagnostics.begin(), result.diagnostics.end(),
                       [&](const auto& diagnostic) { return diagnostic.code == code; });
}

class TemporaryDirectory {
public:
    TemporaryDirectory() {
        static std::atomic<uint64_t> next_id{0};
        std::random_device random;
        const auto timestamp = std::chrono::steady_clock::now().time_since_epoch().count();
        path_ = std::filesystem::temp_directory_path() /
                ("labneura-syntax-" + std::to_string(timestamp) + "-" +
                 std::to_string(random()) + "-" + std::to_string(next_id.fetch_add(1)));
        std::filesystem::create_directories(path_);
    }

    ~TemporaryDirectory() {
        std::error_code ec;
        std::filesystem::remove_all(path_, ec);
    }

    const std::filesystem::path& path() const noexcept { return path_; }

private:
    std::filesystem::path path_;
};

void write_file(const std::filesystem::path& path, std::string_view text) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream file(path, std::ios::binary);
    if (!file)
        throw std::runtime_error("could not create parser test file");
    file.write(text.data(), static_cast<std::streamsize>(text.size()));
    if (!file)
        throw std::runtime_error("could not write parser test file");
}

const VerilogModuleDeclaration& find_module(const VerilogCompilationUnit& unit,
                                           std::string_view name) {
    const auto found = std::find_if(unit.modules.begin(), unit.modules.end(),
                                    [&](const auto& module) { return module.name == name; });
    if (found == unit.modules.end())
        throw std::runtime_error("expected module was not parsed");
    return *found;
}

void test_fixture_and_slang_acceptance() {
    const std::filesystem::path fixture(LABNEURA_SYNTAX_FIXTURE_PATH);
    const auto result = parse_verilog_syntax_file(fixture);
    require(!result.has_errors(), "representative parser fixture should parse cleanly");
    require(result.syntax.modules.size() == 2 && result.syntax.interfaces.size() == 1 &&
                result.syntax.packages.size() == 1,
            "compilation unit declarations were not all retained");

    const auto& child = find_module(result.syntax, "child");
    require(child.parameters.size() == 2 && child.parameters[0].name == "WIDTH" &&
                child.parameters[1].type == "int" &&
                child.ports.size() == 2 && child.assignments.size() == 1,
            "module parameter, ANSI ports, or continuous assignment is missing");
    require(child.ports[0].direction == VerilogPortDirection::Input &&
                child.ports[0].packed_dimensions.size() == 1 &&
                child.ports[0].name == "data",
            "ANSI input direction or packed range is incorrect");

    const auto& top = find_module(result.syntax, "top");
    require(top.ports.size() == 3 && top.ports[0].direction == VerilogPortDirection::Input &&
                top.ports[0].type.empty() &&
                top.ports[1].packed_dimensions.size() == 1 &&
                top.ports[2].direction == VerilogPortDirection::Output,
            "non-ANSI port list shape or body declarations are incorrect");
    require(top.declarations.size() == 2 && top.declarations[0].packed_dimensions.size() == 1 &&
                top.declarations[0].unpacked_dimensions.size() == 1,
            "packed and unpacked variable dimensions were not parsed");
    require(top.assignments.size() == 2 && top.assignments[1].right &&
                top.assignments[1].right->kind == VerilogExpressionKind::Conditional,
            "continuous assignment expression or conditional precedence is incorrect");
    const auto& conditional = *top.assignments[1].right;
    require(conditional.first &&
                conditional.first->kind == VerilogExpressionKind::Binary &&
                conditional.first->text == ">=" &&
                conditional.first->first->kind == VerilogExpressionKind::Binary &&
                conditional.first->first->text == "*",
            "binary precedence in conditional expression was not preserved");
    require(top.instantiations.size() == 2 &&
                top.instantiations[0].instances[0].connections[0].name == "data" &&
                top.instantiations[1].instances[0].connections[0].name == std::nullopt &&
                top.instantiations[0].parameter_overrides[0].named_override &&
                top.instantiations[1].parameter_overrides[0].value,
            "named and positional instance connections or parameter overrides are incorrect");

    const auto& interface = result.syntax.interfaces.front();
    require(interface.name == "stream_if" && interface.parameters.size() == 1 &&
                interface.ports.size() == 2 && interface.declarations.size() == 1,
            "interface parameters, ports, or declarations are missing");
    require(result.syntax.packages.front().name == "parser_types" &&
                result.syntax.packages.front().parameters.size() == 1,
            "package parameter declaration is missing");

    slang::SourceManager source_manager;
    std::ifstream input(fixture, std::ios::binary);
    const std::string source{std::istreambuf_iterator<char>(input),
                             std::istreambuf_iterator<char>()};
    const auto slang_tree = slang::syntax::SyntaxTree::fromText(
        source, source_manager, "parser_foundation.sv");
    require(slang_tree && slang_tree->diagnostics().empty(),
            "Slang syntax parser rejected the supported native fixture");
}

void test_preprocessor_source_mapping_and_includes() {
    TemporaryDirectory temporary;
    const auto include_path = temporary.path() / "ports.vh";
    write_file(include_path, "logic included_signal;\n");
    const auto root_path = temporary.path() / "top.sv";
    const std::string source =
        "`define WIDTH 5\n"
        "module mapped(input logic [`WIDTH-1:0] data);\n"
        "`include \"ports.vh\"\n"
        "endmodule\n";
    write_file(root_path, source);
    VerilogPreprocessorOptions preprocessor_options;
    preprocessor_options.include_paths.push_back(temporary.path());
    auto result = parse_verilog_syntax(source, preprocessor_options, {},
                                       root_path.string());
    require(!result.has_errors() && result.syntax.modules.size() == 1,
            "native parse pipeline failed on macros and included declarations");
    const auto& module = result.syntax.modules.front();
    require(module.ports.size() == 1 && module.ports.front().packed_dimensions.size() == 1 &&
                module.declarations.size() == 1 &&
                module.declarations.front().name == "included_signal",
            "preprocessed port range or included declaration is missing");
    std::error_code ec;
    const auto canonical_include = std::filesystem::weakly_canonical(include_path, ec);
    require(module.declarations.front().location.source &&
                module.declarations.front().location.source->name() ==
                    canonical_include.string() &&
                module.declarations.front().location.location.line == 1,
            "included AST declaration did not map to its original source location");
    const auto& range = module.ports.front().packed_dimensions.front();
    require(range.left && range.left->first &&
                range.left->first->location.origin_kind ==
                              VerilogPreprocessedOriginKind::MacroExpansion &&
                !range.left->first->location.expansion_stack.empty() &&
                range.left->first->location.expansion_stack.back().macro_name == "WIDTH",
            "macro-expanded expression lost its source-map expansion origin");
}

void test_non_ansi_ports_and_expression_forms() {
    const std::string source =
        "module legacy(a, b);\n"
        "input wire [3:0] a;\n"
        "output reg b;\n"
        "wire [3:0] local_value;\n"
        "assign local_value = {a[2:0], 1'b0};\n"
        "assign b = ~|local_value;\n"
        "assign b = bus.ready;\n"
        "endmodule\n";
    const auto result = parse_verilog_syntax(source);
    require(!result.has_errors() && result.syntax.modules.size() == 1,
            "non-ANSI ports or supported expression forms failed to parse");
    const auto& module = result.syntax.modules.front();
    require(module.ports.size() == 2 &&
                module.ports[0].direction == VerilogPortDirection::Input &&
                module.ports[0].type == "wire" &&
                module.ports[1].direction == VerilogPortDirection::Output &&
                module.ports[1].type == "reg",
            "non-ANSI body port declarations were not applied to header ports");
    require(module.assignments.size() == 3 &&
                module.assignments[0].right->kind == VerilogExpressionKind::Concatenation &&
                module.assignments[1].right->kind == VerilogExpressionKind::Unary &&
                module.assignments[1].right->text == "~|" &&
                module.assignments[2].right->kind == VerilogExpressionKind::MemberAccess &&
                module.assignments[2].right->text == "ready",
            "concatenation, select, or unary expression nodes are incorrect");
}

void test_indexed_part_select_ast_modes() {
    const auto parsed = parse_verilog_syntax(
        "module indexed_selects;\n"
        "task automatic update(input integer base);\n"
        "  reg [7:0] value;\n"
        "  begin value[base +: 3] = 3'b101; value[base -: 3] = 3'b011; end\n"
        "endtask\n"
        "endmodule\n");
    require(!parsed.has_errors() && parsed.syntax.modules.size() == 1,
            "indexed part-select assignments should parse without errors");
    const auto& task = parsed.syntax.modules.front().callables.front();
    std::vector<const VerilogExpression*> selects;
    const auto collect = [&](const auto& self, const VerilogStatement* statement) -> void {
        if (!statement)
            return;
        if (statement->kind == VerilogStatementKind::ProceduralAssignment &&
            statement->left &&
            statement->left->kind == VerilogExpressionKind::Select)
            selects.push_back(statement->left.get());
        self(self, statement->first_statement.get());
        self(self, statement->second_statement.get());
        for (const auto& nested : statement->statements)
            self(self, nested.get());
    };
    collect(collect, task.body.get());
    require(selects.size() == 2 && selects[0]->third && selects[1]->third &&
                selects[0]->select_kind == VerilogSelectKind::IndexedUp &&
                selects[1]->select_kind == VerilogSelectKind::IndexedDown,
            "indexed part-select direction was not retained in the AST");
}

void test_malformed_syntax_recovery_and_locations() {
    const std::string source =
        "module broken(input logic a) wire missing_semicolon;\n"
        "endmodule\n"
        "module valid;\n"
        "logic okay;\n"
        "assign okay = 1'b1;\n"
        "endmodule\n";
    const auto result = parse_verilog_syntax(source);
    require(result.has_errors() && result.syntax.modules.size() == 2 &&
                result.syntax.modules.back().name == "valid",
            "malformed module did not produce diagnostics and recover at endmodule");
    const auto expected = std::find_if(
        result.diagnostics.begin(), result.diagnostics.end(), [](const auto& diagnostic) {
            return diagnostic.code == "ExpectedToken";
        });
    require(expected != result.diagnostics.end() && expected->location.source &&
                expected->location.location.line == 1,
            "expected-token diagnostic has incorrect original source location");
    require(result.syntax.modules.back().declarations.front().location.location.line == 4,
            "AST node location is not line-accurate");

    const auto missing_end = parse_verilog_syntax(
        "module unfinished;\nlogic value;\n"
        "module survivor;\nendmodule\n");
    require(has_error_code(missing_end, "MissingEndmodule") &&
                missing_end.syntax.modules.size() == 2 &&
                missing_end.syntax.modules.back().name == "survivor",
            "parser did not recover at the next design unit after missing endmodule");
}

void test_behavioral_rtl_and_slang_acceptance() {
    const std::filesystem::path fixture(LABNEURA_BEHAVIORAL_FIXTURE_PATH);
    auto result = parse_verilog_syntax_file(fixture);
    require(!result.has_errors() && result.syntax.modules.size() == 1,
            "behavioral RTL fixture should parse without diagnostics");
    const auto& module = find_module(result.syntax, "behavioral_rtl");
    require(module.procedural_blocks.size() == 5,
            "always, always_comb, always_ff, always_latch, and initial blocks were not retained");

    const auto& combinational = module.procedural_blocks[0];
    require(combinational.kind == VerilogProceduralBlockKind::AlwaysComb &&
                combinational.body &&
                combinational.body->kind == VerilogStatementKind::Block &&
                combinational.body->label == "comb" &&
                combinational.body->statements.size() == 7,
            "always_comb body or labeled sequential block was not parsed");
    const auto& part_assignment = *combinational.body->statements[0];
    require(part_assignment.kind == VerilogStatementKind::ProceduralAssignment &&
                part_assignment.assignment_kind == VerilogAssignmentKind::Blocking &&
                part_assignment.left &&
                part_assignment.left->kind == VerilogExpressionKind::Select &&
                part_assignment.left->third,
            "blocking assignment or part-select target was not retained");

    require(combinational.body->statements[2]->left &&
                combinational.body->statements[2]->left->kind ==
                    VerilogExpressionKind::Concatenation,
            "concatenation assignment target was not retained");

    const auto& conditional = *combinational.body->statements[3];
    require(conditional.kind == VerilogStatementKind::If &&
                conditional.first_statement &&
                conditional.first_statement->kind == VerilogStatementKind::Block &&
                conditional.first_statement->label == "selected" &&
                conditional.second_statement &&
                conditional.second_statement->kind == VerilogStatementKind::If &&
                conditional.second_statement->second_statement,
            "nested if/else-if/else or block labels were not parsed");
    const auto& macro_assignment =
        *conditional.first_statement->statements.front();
    require(macro_assignment.right &&
                macro_assignment.right->location.origin_kind ==
                    VerilogPreprocessedOriginKind::MacroExpansion &&
                !macro_assignment.right->location.expansion_stack.empty(),
            "procedural macro expansion lost its source origin");

    const auto& plain_case = *combinational.body->statements[4];
    const auto& casez = *combinational.body->statements[5];
    const auto& casex = *combinational.body->statements[6];
    require(plain_case.kind == VerilogStatementKind::Case &&
                plain_case.case_kind == VerilogCaseKind::Case &&
                plain_case.case_items.front().expressions.size() == 2 &&
                plain_case.case_items.back().is_default &&
                casez.case_kind == VerilogCaseKind::CaseZ &&
                casex.case_kind == VerilogCaseKind::CaseX,
            "case, casez, casex, multiple labels, or default item were not parsed");

    const auto& clocked = module.procedural_blocks[1];
    require(clocked.kind == VerilogProceduralBlockKind::AlwaysFF &&
                clocked.event_control && clocked.event_control->terms.size() == 2 &&
                clocked.event_control->terms[0].edge == VerilogEventEdge::Posedge &&
                clocked.event_control->terms[1].edge == VerilogEventEdge::Negedge &&
                clocked.body && clocked.body->kind == VerilogStatementKind::Block,
            "always_ff edge event list or body was not parsed");
    const auto& clocked_if = *clocked.body->statements.front();
    require(clocked_if.kind == VerilogStatementKind::If &&
                clocked_if.first_statement &&
                clocked_if.first_statement->assignment_kind ==
                    VerilogAssignmentKind::Nonblocking &&
                clocked_if.second_statement &&
                clocked_if.second_statement->kind == VerilogStatementKind::If,
            "active-low reset and nested nonblocking assignments were not retained");

    const auto& latch_block = module.procedural_blocks[2];
    require(latch_block.kind == VerilogProceduralBlockKind::AlwaysLatch &&
                latch_block.body && latch_block.body->kind == VerilogStatementKind::Block,
            "always_latch body was not parsed");

    const auto& edge_block = module.procedural_blocks[3];
    require(edge_block.kind == VerilogProceduralBlockKind::Always &&
                edge_block.event_control && edge_block.event_control->terms.size() == 2 &&
                edge_block.body && edge_block.body->kind == VerilogStatementKind::If,
            "always event control with 'or' separator was not parsed");

    const auto& initial = module.procedural_blocks[4];
    require(initial.kind == VerilogProceduralBlockKind::Initial && initial.body &&
                initial.body->statements.size() == 5 &&
                initial.body->statements[0]->kind == VerilogStatementKind::DelayControl &&
                initial.body->statements[0]->expression &&
                initial.body->statements[1]->kind == VerilogStatementKind::EventControl &&
                initial.body->statements[2]->kind == VerilogStatementKind::EventControl &&
                initial.body->statements[3]->kind == VerilogStatementKind::EventControl &&
                initial.body->statements[1]->event_control &&
                initial.body->statements[2]->event_control->wildcard &&
                initial.body->statements[3]->event_control->wildcard &&
                initial.body->statements[4]->kind == VerilogStatementKind::Wait,
            "initial delay, event controls, wildcard controls, or wait were not parsed");

    slang::SourceManager source_manager;
    std::ifstream input(fixture, std::ios::binary);
    const std::string source{std::istreambuf_iterator<char>(input),
                             std::istreambuf_iterator<char>()};
    const auto slang_tree =
        slang::syntax::SyntaxTree::fromText(source, source_manager, "behavioral_rtl.sv");
    require(slang_tree && slang_tree->diagnostics().empty(),
            "Slang syntax parser rejected the supported behavioral RTL fixture");
}

void test_procedural_loops_and_typed_controls() {
    const std::string source =
        "module procedural_loops(input logic clk, input logic condition, "
        "output logic [7:0] q);\n"
        "  integer i;\n"
        "  always @(posedge clk) begin : outer\n"
        "    for (i = 0; i < 4; i = i + 1) begin : for_block\n"
        "      q[i] <= i;\n"
        "    end\n"
        "    while (condition) q = q + 1;\n"
        "    repeat (2) begin q = q - 1; end\n"
        "    for (int j = 0; j < 2; j++) q[j] = j;\n"
        "    forever begin : forever_block @(posedge clk) q = q; end\n"
        "  end\n"
        "  initial begin #5 q = '0; end\n"
        "endmodule\n";
    const auto parsed = parse_verilog_syntax(source);
    if (parsed.has_errors()) {
        std::string messages;
        for (const auto& diagnostic : parsed.diagnostics)
            messages += (messages.empty() ? "" : "; ") + diagnostic.code + ": " +
                        diagnostic.message;
        throw std::runtime_error("valid procedural loops failed to parse: " + messages);
    }

    const auto& module = find_module(parsed.syntax, "procedural_loops");
    const auto& block = *module.procedural_blocks.front().body;
    require(block.kind == VerilogStatementKind::Block && block.label == "outer" &&
                block.statements.size() == 5,
            "outer procedural block or loop count is incorrect");
    const auto& for_loop = *block.statements[0];
    require(for_loop.kind == VerilogStatementKind::Loop &&
                for_loop.loop_kind == VerilogLoopKind::For &&
                for_loop.loop_initialization && for_loop.expression && for_loop.loop_step &&
                for_loop.first_statement &&
                for_loop.first_statement->kind == VerilogStatementKind::Block &&
                for_loop.first_statement->label == "for_block" &&
                for_loop.loop_step->right &&
                for_loop.loop_step->right->kind == VerilogExpressionKind::Binary,
            "for-loop initialization, condition, update, or named body is missing");
    require(block.statements[1]->kind == VerilogStatementKind::Loop &&
                block.statements[1]->loop_kind == VerilogLoopKind::While &&
                block.statements[1]->expression &&
                block.statements[2]->loop_kind == VerilogLoopKind::Repeat &&
                block.statements[2]->expression &&
                block.statements[3]->loop_kind == VerilogLoopKind::For &&
                block.statements[3]->loop_variable_name == "j" &&
                block.statements[3]->loop_variable_type == "int" &&
                block.statements[3]->loop_variable_location.location.line == 9 &&
                block.statements[3]->loop_step->right &&
                block.statements[4]->loop_kind == VerilogLoopKind::Forever,
            "while, repeat, declared for-variable, or forever loop was not retained");
    require(block.statements[4]->first_statement &&
                block.statements[4]->first_statement->kind ==
                    VerilogStatementKind::Block &&
                block.statements[4]->first_statement->statements.front()->kind ==
                    VerilogStatementKind::EventControl &&
                module.procedural_blocks.back().body->statements.front()->kind ==
                    VerilogStatementKind::DelayControl,
            "nested event and delay controls do not have explicit statement kinds");

    const auto compound_update = parse_verilog_syntax(
        "module compound_update;\n"
        "  integer i;\n"
        "  initial for (i = 1; i < 8; i <<= 1) ;\n"
        "endmodule\n");
    const auto& compound_loop =
        *compound_update.syntax.modules.front().procedural_blocks.front().body;
    require(!compound_update.has_errors() && compound_loop.loop_step &&
                compound_loop.loop_step->right &&
                compound_loop.loop_step->right->text == "<<",
            "shift compound loop update was not retained");

    slang::SourceManager source_manager;
    const auto slang_tree =
        slang::syntax::SyntaxTree::fromText(source, source_manager, "procedural_loops.sv");
    require(slang_tree && slang_tree->diagnostics().empty(),
            "Slang syntax parser rejected the supported procedural-loop fixture");

    const auto unsupported = parse_verilog_syntax(
        "module unsupported_continuous(input logic a, output logic q);\n"
        "  always begin assign q = a; deassign q; force q = a; release q; end\n"
        "endmodule\n");
    require(has_error_code(unsupported, "UnsupportedProceduralStatement") &&
                unsupported.syntax.modules.front().procedural_blocks.front()
                        .body->statements.size() == 4 &&
                std::all_of(
                    unsupported.syntax.modules.front().procedural_blocks.front().body->statements
                        .begin(),
                    unsupported.syntax.modules.front().procedural_blocks.front().body->statements
                        .end(),
                    [](const auto& statement) {
                        return statement->kind == VerilogStatementKind::Unsupported;
                    }),
            "procedural continuous forms were not explicitly represented as unsupported");

    const auto malformed = parse_verilog_syntax(
        "module malformed_loop;\n"
        "  integer i; logic q;\n"
        "  initial begin for (i = 0 i < 2; i++) q = 1; end\n"
        "endmodule\n"
        "module after_malformed_loop; endmodule\n");
    require(malformed.has_errors() && malformed.syntax.modules.size() == 2 &&
                malformed.syntax.modules.back().name == "after_malformed_loop" &&
                has_error_code(malformed, "MalformedForLoop"),
            "malformed loop diagnostics prevented recovery to the next module");

    const auto missing_step = parse_verilog_syntax(
        "module missing_step;\n"
        "  integer i; logic q;\n"
        "  initial for (i = 0; i < 2; i +=) q = 1;\n"
        "endmodule\n");
    require(has_error_code(missing_step, "MissingForStep"),
            "compound for-loop update without a value was accepted");
}

void test_do_while_and_loop_control_ast() {
    const auto parsed = parse_verilog_syntax(
        "module loop_control;\n"
        "initial begin\n"
        "  do begin break; continue; end while (flag);\n"
        "end\n"
        "endmodule\n");
    require(!parsed.has_errors(), "valid do-while/control statements failed to parse");
    const auto& body =
        *parsed.syntax.modules.front().procedural_blocks.front().body;
    const auto& loop = *body.statements.front();
    require(loop.kind == VerilogStatementKind::Loop &&
                loop.loop_kind == VerilogLoopKind::DoWhile && loop.expression &&
                loop.first_statement &&
                loop.first_statement->statements.size() == 2 &&
                loop.first_statement->statements[0]->kind ==
                    VerilogStatementKind::Break &&
                loop.first_statement->statements[1]->kind ==
                    VerilogStatementKind::Continue,
            "do-while or loop-control statements were not retained in the AST");
}

void test_bare_signal_event_controls_and_semantic_resolution() {
    const auto parsed = parse_verilog_syntax(
        "module bare_signal_events;\n"
        "  logic signal;\n"
        "  logic result;\n"
        "  always @signal result = 1'b1;\n"
        "  initial begin @signal result = 1'b0; end\n"
        "endmodule\n");
    require(!parsed.has_errors(), "bare @signal event controls should parse cleanly");
    // Slang is used here only to confirm that the fixture is valid syntax;
    // no procedural timing or simulation equivalence is inferred.
    slang::SourceManager source_manager;
    const auto slang_tree = slang::syntax::SyntaxTree::fromText(
        "module bare_signal_events; logic signal, result;\n"
        "  always @signal result = 1'b1;\n"
        "  initial begin @signal result = 1'b0; end\n"
        "endmodule\n",
        source_manager, "bare_signal_events.sv");
    require(slang_tree && slang_tree->diagnostics().empty(),
            "Slang rejected the supported bare @signal syntax fixture");
    const auto& module = parsed.syntax.modules.front();
    require(module.procedural_blocks.size() == 2,
            "bare @signal procedural blocks were not retained");

    const auto& always_control = module.procedural_blocks[0].event_control;
    require(always_control && !always_control->wildcard &&
                always_control->terms.size() == 1 &&
                always_control->terms.front().edge == VerilogEventEdge::Any &&
                always_control->terms.front().expression &&
                always_control->terms.front().expression->kind ==
                    VerilogExpressionKind::Identifier &&
                always_control->terms.front().expression->text == "signal" &&
                always_control->location.location.line == 4 &&
                always_control->terms.front().location.location.line == 4,
            "bare @signal in an always block lost its level event or source location");

    const auto& initial_body = module.procedural_blocks[1].body;
    const auto* initial_event =
        initial_body && !initial_body->statements.empty()
            ? initial_body->statements.front().get()
            : nullptr;
    require(initial_event && initial_event->kind == VerilogStatementKind::EventControl &&
                initial_event->event_control &&
                initial_event->event_control->terms.size() == 1 &&
                initial_event->event_control->terms.front().edge == VerilogEventEdge::Any &&
                initial_event->event_control->terms.front().expression &&
                initial_event->event_control->terms.front().expression->text == "signal" &&
                initial_event->event_control->location.location.line == 5,
            "bare @signal statement lost its event term or source location");

    const auto semantic = analyze_verilog_semantics(parsed.syntax);
    require(!semantic.has_errors(), "bare event signal references should resolve");
    for (const auto* expression : {
             always_control->terms.front().expression.get(),
             initial_event->event_control->terms.front().expression.get()}) {
        const auto reference = std::find_if(
            semantic.references.begin(), semantic.references.end(),
            [&](const auto& item) {
                return item.location.byte_offset == expression->location.byte_offset &&
                       item.location.source == expression->location.source;
            });
        require(reference != semantic.references.end() && reference->declaration,
                "semantic analysis did not resolve the bare event signal");
    }
}

void test_malformed_procedural_recovery_and_unsupported_statement() {
    const auto malformed = parse_verilog_syntax(
        "module broken;\n"
        "always_comb begin\n"
        "  if () q <= ;\n"
        "endmodule\n"
        "module survivor;\n"
        "initial begin q = 1; end\n"
        "endmodule\n");
    require(malformed.has_errors() && malformed.syntax.modules.size() == 2 &&
                malformed.syntax.modules.back().name == "survivor" &&
                has_error_code(malformed, "MissingIfCondition") &&
                has_error_code(malformed, "MissingAssignmentExpression") &&
                has_error_code(malformed, "ExpectedEnd"),
            "malformed procedural syntax did not diagnose and recover into the next module");

    const auto unsupported = parse_verilog_syntax(
        "module unsupported;\n"
        "always_comb begin\n"
        "  assert (1'b1);\n"
        "end\n"
        "endmodule\n");
    require(has_error_code(unsupported, "UnsupportedProceduralStatement") &&
                unsupported.syntax.modules.size() == 1 &&
                unsupported.syntax.modules.front().procedural_blocks.front()
                        .body->statements.front()->kind == VerilogStatementKind::Unsupported,
            "unsupported procedural syntax was silently discarded or swallowed the module");

    const auto malformed_event = parse_verilog_syntax(
        "module bad;\n"
        "always @(posedge clk begin q <= 1; end\n"
        "endmodule\n"
        "module after_bad_event; endmodule\n");
    require(has_error_code(malformed_event, "MalformedEventControl") &&
                malformed_event.syntax.modules.size() == 2 &&
                malformed_event.syntax.modules.back().name == "after_bad_event",
            "malformed event-control delimiter was not diagnosed or recovered");

    const auto malformed_case = parse_verilog_syntax(
        "module bad_case;\n"
        "always_comb case (flag q = 1;\n"
        "endmodule\n"
        "module after_bad_case; endmodule\n");
    require(has_error_code(malformed_case, "MalformedCaseStatement") &&
                has_error_code(malformed_case, "ExpectedEndcase") &&
                malformed_case.syntax.modules.size() == 2 &&
                malformed_case.syntax.modules.back().name == "after_bad_case",
            "malformed case delimiters were not diagnosed or recovered");
}

} // namespace

int main() {
    try {
        test_fixture_and_slang_acceptance();
        test_preprocessor_source_mapping_and_includes();
        test_non_ansi_ports_and_expression_forms();
        test_indexed_part_select_ast_modes();
        test_malformed_syntax_recovery_and_locations();
        test_behavioral_rtl_and_slang_acceptance();
        test_procedural_loops_and_typed_controls();
        test_do_while_and_loop_control_ast();
        test_malformed_procedural_recovery_and_unsupported_statement();
        test_bare_signal_event_controls_and_semantic_resolution();
    }
    catch (const std::exception& error) {
        std::cerr << "EDA Verilog syntax parser test failed: " << error.what() << '\n';
        return 1;
    }
}
