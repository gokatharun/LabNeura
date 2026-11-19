#include "labneura/eda/verilog_semantic.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using namespace labneura::eda;
constexpr std::size_t kMissing = std::numeric_limits<std::size_t>::max();

struct Key {
    const VerilogSourceBuffer* source;
    std::size_t offset;
    friend bool operator==(const Key&, const Key&) = default;
};

struct Hash {
    std::size_t operator()(const Key& key) const noexcept {
        return (reinterpret_cast<std::uintptr_t>(key.source) >> 4) ^ key.offset;
    }
};

Key key(const VerilogSemanticReference& reference) {
    return {reference.location.source.get(), reference.location.byte_offset};
}

std::size_t legacy(const std::vector<VerilogSemanticReference>& references,
                   Key sought) {
    for (std::size_t index = 0; index < references.size(); ++index) {
        if (references[index].declaration && key(references[index]) == sought)
            return index;
    }
    return kMissing;
}

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

} // namespace

int main() {
    try {
        auto first_source = std::make_shared<const VerilogSourceBuffer>("abc", "same.sv");
        auto second_source = std::make_shared<const VerilogSourceBuffer>("abc", "same.sv");
        VerilogSemanticSymbol outer;
        outer.name = "value";
        VerilogSemanticSymbol inner;
        inner.name = "value";
        VerilogSemanticSymbol parameter;
        parameter.name = "WIDTH";

        std::vector<VerilogSemanticReference> references = {
            {"value", {.source = first_source, .byte_offset = 1}, nullptr, &outer},
            {"value", {.source = first_source, .byte_offset = 1}, nullptr, &inner},
            {"value", {.source = second_source, .byte_offset = 1}, nullptr, &inner},
            {"WIDTH", {.source = first_source, .byte_offset = 2}, nullptr, &parameter},
            {"missing", {.source = first_source, .byte_offset = 3}, nullptr, nullptr},
        };

        std::unordered_map<Key, std::size_t, Hash> index;
        for (std::size_t position = 0; position < references.size(); ++position) {
            if (references[position].declaration)
                index.emplace(key(references[position]), position);
        }

        const std::vector<Key> queries = {
            {first_source.get(), 1}, {second_source.get(), 1},
            {first_source.get(), 2}, {first_source.get(), 3},
            {first_source.get(), 99},
        };
        for (const auto& query : queries) {
            const auto found = index.find(query);
            const auto actual = found == index.end() ? kMissing : found->second;
            require(actual == legacy(references, query),
                    "hash experiment diverged from the linear oracle");
        }
        require(index.at({first_source.get(), 1}) == 0,
                "duplicate key did not retain first-match behavior");
        require(index.at({second_source.get(), 1}) == 2,
                "distinct source buffers with identical positions collided");

        auto copied = index;
        auto moved = std::move(copied);
        require(moved.at({first_source.get(), 2}) == 3,
                "copy/move changed indexed reference identity");
        std::cout << "semantic reference index experiment tests passed\n";
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
