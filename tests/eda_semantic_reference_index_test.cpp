#include "labneura/eda/verilog_semantic.h"
#include "labneura/eda/verilog_syntax.h"

#include <algorithm>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

using namespace labneura::eda;

void require(bool condition, std::string_view message) {
    if (!condition)
        throw std::runtime_error(std::string(message));
}

std::vector<size_t> linear_ordinals(
    const VerilogSemanticAnalysisResult& semantic,
    const VerilogSyntaxLocation& location) {
    std::vector<size_t> result;
    for (size_t ordinal = 0; ordinal < semantic.references.size(); ++ordinal) {
        const auto& reference = semantic.references[ordinal];
        if (reference.location.source == location.source &&
            reference.location.byte_offset == location.byte_offset)
            result.push_back(ordinal);
    }
    return result;
}

std::vector<size_t> indexed_ordinals(
    const VerilogSemanticAnalysisResult& semantic,
    const VerilogSyntaxLocation& location) {
    std::vector<size_t> result;
    for (const auto& entry : semantic.references_at(location))
        result.push_back(entry.reference_ordinal);
    return result;
}

void test_real_analysis_matches_linear_oracle() {
    const auto parsed = parse_verilog_syntax(R"(
module indexed #(parameter int WIDTH = 4) (input logic a, output logic y);
  logic value;
  assign value = a;
  assign y = value;
endmodule
)");
    require(!parsed.has_errors(), "index fixture should parse");
    const auto semantic = analyze_verilog_semantics(parsed.syntax);
    require(!semantic.has_errors(), "index fixture should resolve");
    require(semantic.reference_index_statistics.reference_count ==
                semantic.references.size(),
            "index statistics reference count is stale");
    require(semantic.reference_index_statistics.entry_count ==
                semantic.references.size(),
            "every reference must have one index entry");

    for (const auto& reference : semantic.references)
        require(indexed_ordinals(semantic, reference.location) ==
                    linear_ordinals(semantic, reference.location),
                "indexed lookup diverged from the linear oracle");

    auto missing = semantic.references.front().location;
    missing.byte_offset = parsed.syntax.location.source->text().size() + 1;
    require(semantic.references_at(missing).empty(),
            "missing location should return an empty range");
}

void test_duplicates_source_identity_rebuild_and_move() {
    auto first = std::make_shared<const VerilogSourceBuffer>("abc", "same.sv");
    auto second = std::make_shared<const VerilogSourceBuffer>("abc", "same.sv");
    VerilogSemanticSymbol outer;
    VerilogSemanticSymbol inner;
    VerilogSemanticAnalysisResult semantic;
    semantic.references = {
        {"value", {.source = first, .byte_offset = 1}, nullptr, nullptr},
        {"value", {.source = first, .byte_offset = 1}, nullptr, &outer},
        {"value", {.source = second, .byte_offset = 1}, nullptr, &inner},
        {"value", {.source = first, .byte_offset = 1}, nullptr, &inner},
    };
    semantic.rebuild_reference_index();

    const VerilogSyntaxLocation first_location{.source = first, .byte_offset = 1};
    const VerilogSyntaxLocation second_location{.source = second, .byte_offset = 1};
    require(indexed_ordinals(semantic, first_location) ==
                std::vector<size_t>({0, 1, 3}),
            "duplicates must retain insertion order including null declarations");
    require(indexed_ordinals(semantic, second_location) ==
                std::vector<size_t>({2}),
            "buffer identity must distinguish equal source contents");
    require(semantic.reference_index_statistics.duplicate_location_count == 2,
            "duplicate statistics should count entries beyond the first");

    const auto original_index = semantic.reference_index;
    semantic.rebuild_reference_index();
    require(semantic.reference_index.size() == original_index.size() &&
                std::equal(semantic.reference_index.begin(),
                           semantic.reference_index.end(), original_index.begin(),
                           [](const auto& left, const auto& right) {
                               return left.source == right.source &&
                                      left.byte_offset == right.byte_offset &&
                                      left.reference_ordinal == right.reference_ordinal;
                           }),
            "rebuilding the index must be deterministic");

    auto moved = std::move(semantic);
    require(indexed_ordinals(moved, first_location) ==
                std::vector<size_t>({0, 1, 3}),
            "moving a semantic result must preserve index identity");
}

} // namespace

int main() {
    try {
        test_real_analysis_matches_linear_oracle();
        test_duplicates_source_identity_rebuild_and_move();
        std::cout << "semantic reference index tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
