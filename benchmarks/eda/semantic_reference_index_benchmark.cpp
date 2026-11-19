#include "labneura/eda/verilog_semantic.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using labneura::eda::VerilogSemanticReference;
using labneura::eda::VerilogSemanticSymbol;
using labneura::eda::VerilogSourceBuffer;

constexpr std::uint64_t kSeed = 0x4d37313832ULL;
constexpr std::size_t kMissing = std::numeric_limits<std::size_t>::max();

struct LocationKey {
    const VerilogSourceBuffer* source = nullptr;
    std::size_t offset = 0;

    friend bool operator==(const LocationKey&, const LocationKey&) = default;
};

struct NamedKey {
    LocationKey location;
    std::string name;

    friend bool operator==(const NamedKey&, const NamedKey&) = default;
};

struct LocationHash {
    std::size_t operator()(const LocationKey& key) const noexcept {
        const auto pointer = reinterpret_cast<std::uintptr_t>(key.source);
        return (pointer >> 4) ^ (key.offset + 0x9e3779b97f4a7c15ULL +
                                 (pointer << 6) + (pointer >> 2));
    }
};

struct NamedHash {
    std::size_t operator()(const NamedKey& key) const noexcept {
        return LocationHash{}(key.location) ^
               (std::hash<std::string>{}(key.name) + 0x9e3779b97f4a7c15ULL);
    }
};

bool location_less(const LocationKey& left, const LocationKey& right) {
    const auto less = std::less<const VerilogSourceBuffer*>{};
    if (less(left.source, right.source)) return true;
    if (less(right.source, left.source)) return false;
    return left.offset < right.offset;
}

LocationKey location_key(const VerilogSemanticReference& reference) {
    return {reference.location.source.get(), reference.location.byte_offset};
}

std::size_t linear_location(const std::vector<VerilogSemanticReference>& refs,
                            const LocationKey& key) {
    for (std::size_t index = 0; index < refs.size(); ++index) {
        if (location_key(refs[index]) == key && refs[index].declaration)
            return index;
    }
    return kMissing;
}

std::size_t linear_named(const std::vector<VerilogSemanticReference>& refs,
                         const NamedKey& key) {
    for (std::size_t index = 0; index < refs.size(); ++index) {
        if (location_key(refs[index]) == key.location &&
            refs[index].name == key.name && refs[index].declaration)
            return index;
    }
    return kMissing;
}

class SortedLocationIndex {
public:
    explicit SortedLocationIndex(const std::vector<VerilogSemanticReference>& refs) {
        entries_.reserve(refs.size());
        for (std::size_t index = 0; index < refs.size(); ++index) {
            if (refs[index].declaration)
                entries_.push_back({location_key(refs[index]), index});
        }
        std::stable_sort(entries_.begin(), entries_.end(), [](const auto& left,
                                                               const auto& right) {
            return location_less(left.first, right.first);
        });
    }

    std::size_t find(const LocationKey& key) const {
        const auto found = std::lower_bound(
            entries_.begin(), entries_.end(), key,
            [](const auto& entry, const auto& sought) {
                return location_less(entry.first, sought);
            });
        return found != entries_.end() && found->first == key ? found->second
                                                              : kMissing;
    }

    std::size_t storage_bytes() const { return entries_.capacity() * sizeof(entries_[0]); }

private:
    std::vector<std::pair<LocationKey, std::size_t>> entries_;
};

class HashLocationIndex {
public:
    explicit HashLocationIndex(const std::vector<VerilogSemanticReference>& refs) {
        entries_.reserve(refs.size());
        // emplace preserves the first declaration-bearing reference for a key.
        for (std::size_t index = 0; index < refs.size(); ++index) {
            if (refs[index].declaration)
                entries_.emplace(location_key(refs[index]), index);
        }
    }

    std::size_t find(const LocationKey& key) const {
        const auto found = entries_.find(key);
        return found == entries_.end() ? kMissing : found->second;
    }

    std::size_t storage_bytes() const {
        return entries_.bucket_count() * sizeof(void*) +
               entries_.size() * (sizeof(LocationKey) + sizeof(std::size_t) +
                                  2 * sizeof(void*));
    }

private:
    std::unordered_map<LocationKey, std::size_t, LocationHash> entries_;
};

class DenseLocationIndex {
public:
    explicit DenseLocationIndex(const std::vector<VerilogSemanticReference>& refs) {
        for (std::size_t index = 0; index < refs.size(); ++index) {
            if (!refs[index].declaration) continue;
            auto& offsets = sources_[refs[index].location.source.get()];
            const auto offset = refs[index].location.byte_offset;
            if (offset >= offsets.size()) offsets.resize(offset + 1, kMissing);
            if (offsets[offset] == kMissing) offsets[offset] = index;
        }
    }

    std::size_t find(const LocationKey& key) const {
        const auto source = sources_.find(key.source);
        if (source == sources_.end() || key.offset >= source->second.size())
            return kMissing;
        return source->second[key.offset];
    }

    std::size_t storage_bytes() const {
        std::size_t bytes = sources_.bucket_count() * sizeof(void*);
        for (const auto& [source, offsets] : sources_) {
            (void)source;
            bytes += offsets.capacity() * sizeof(std::size_t) + 3 * sizeof(void*);
        }
        return bytes;
    }

private:
    std::unordered_map<const VerilogSourceBuffer*, std::vector<std::size_t>> sources_;
};

class HashNamedIndex {
public:
    explicit HashNamedIndex(const std::vector<VerilogSemanticReference>& refs) {
        entries_.reserve(refs.size());
        for (std::size_t index = 0; index < refs.size(); ++index) {
            if (refs[index].declaration) {
                entries_.emplace(
                    NamedKey{location_key(refs[index]), refs[index].name}, index);
            }
        }
    }

    std::size_t find(const NamedKey& key) const {
        const auto found = entries_.find(key);
        return found == entries_.end() ? kMissing : found->second;
    }

private:
    std::unordered_map<NamedKey, std::size_t, NamedHash> entries_;
};

struct Fixture {
    std::vector<std::shared_ptr<const VerilogSourceBuffer>> sources;
    std::vector<std::unique_ptr<VerilogSemanticSymbol>> symbols;
    std::vector<VerilogSemanticReference> references;
};

Fixture make_fixture(std::size_t count) {
    Fixture fixture;
    for (std::size_t source = 0; source < 3; ++source) {
        fixture.sources.push_back(std::make_shared<const VerilogSourceBuffer>(
            std::string(count * 4 + 64, 'x'), "source" + std::to_string(source)));
    }
    fixture.symbols.reserve(count);
    fixture.references.reserve(count + count / 128 + 1);
    for (std::size_t index = 0; index < count; ++index) {
        auto symbol = std::make_unique<VerilogSemanticSymbol>();
        symbol->name = "signal_" + std::to_string(index % 257);
        const auto* declaration = symbol.get();
        fixture.symbols.push_back(std::move(symbol));
        const auto source_index = index % fixture.sources.size();
        const auto offset = index * 3 + source_index;
        fixture.references.push_back(
            {fixture.symbols.back()->name,
             {.source = fixture.sources[source_index], .byte_offset = offset},
             nullptr, declaration});
        if (index % 128 == 0) {
            // A later duplicate must never replace the legacy first match.
            fixture.references.push_back(fixture.references.back());
        }
    }
    return fixture;
}

std::vector<LocationKey> make_queries(const Fixture& fixture, bool random_order) {
    const auto count = std::min<std::size_t>(fixture.references.size() * 2, 2048);
    std::vector<LocationKey> queries;
    queries.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        if (index % 4 == 3) {
            queries.push_back({fixture.sources[index % fixture.sources.size()].get(),
                               fixture.references.size() * 8 + index});
        }
        else {
            queries.push_back(location_key(
                fixture.references[(index * 37) % fixture.references.size()]));
        }
    }
    if (random_order) {
        std::mt19937_64 random(kSeed);
        std::shuffle(queries.begin(), queries.end(), random);
    }
    return queries;
}

template <typename Function>
double seconds(Function&& function) {
    const auto start = Clock::now();
    function();
    const auto stop = Clock::now();
    return std::chrono::duration<double>(stop - start).count();
}

template <typename Builder, typename Lookup>
void measure_index(std::string_view name, const Fixture& fixture,
                   const std::vector<LocationKey>& queries, Builder&& builder,
                   Lookup&& lookup, std::size_t storage_bytes) {
    std::size_t checksum = 0;
    const double construction = seconds([&] { builder(); });
    const double lookups = seconds([&] {
        for (const auto& query : queries) checksum ^= lookup(query);
    });
    std::cout << std::setprecision(12) << "strategy=" << name
              << " references=" << fixture.references.size()
              << " queries=" << queries.size()
              << " construction_seconds=" << construction
              << " lookup_seconds=" << lookups
              << " total_seconds=" << construction + lookups
              << " estimated_storage_bytes=" << storage_bytes
              << " checksum=" << checksum << '\n';
}

void run(std::size_t count, bool random_order) {
    const auto fixture = make_fixture(count);
    const auto queries = make_queries(fixture, random_order);
    std::cout << "access=" << (random_order ? "random" : "sequential")
              << " seed=" << kSeed << '\n';

    measure_index("linear", fixture, queries, [] {},
                  [&](const auto& key) {
                      return linear_location(fixture.references, key);
                  }, fixture.references.capacity() * sizeof(VerilogSemanticReference));

    std::unique_ptr<SortedLocationIndex> sorted;
    measure_index("sorted", fixture, queries,
                  [&] { sorted = std::make_unique<SortedLocationIndex>(fixture.references); },
                  [&](const auto& key) { return sorted->find(key); },
                  fixture.references.size() * sizeof(std::pair<LocationKey, std::size_t>));

    std::unique_ptr<HashLocationIndex> hashed;
    measure_index("hash", fixture, queries,
                  [&] { hashed = std::make_unique<HashLocationIndex>(fixture.references); },
                  [&](const auto& key) { return hashed->find(key); },
                  fixture.references.size() *
                      (sizeof(LocationKey) + sizeof(std::size_t) + 2 * sizeof(void*)));

    std::unique_ptr<DenseLocationIndex> dense;
    measure_index("dense", fixture, queries,
                  [&] { dense = std::make_unique<DenseLocationIndex>(fixture.references); },
                  [&](const auto& key) { return dense->find(key); },
                  count * 3 * sizeof(std::size_t));

    // Exercise the name-qualified constant-resolution key independently.
    HashNamedIndex named(fixture.references);
    std::size_t named_checksum = 0;
    const double named_lookup = seconds([&] {
        for (const auto& query : queries) {
            const auto legacy = linear_location(fixture.references, query);
            const std::string name = legacy == kMissing
                                         ? "missing"
                                         : fixture.references[legacy].name;
            named_checksum ^= named.find({query, name});
        }
    });
    std::cout << "strategy=hash_named references=" << fixture.references.size()
              << " queries=" << queries.size()
              << " lookup_seconds=" << named_lookup
              << " checksum=" << named_checksum << '\n';
}

} // namespace

int main() {
    for (const auto count : {std::size_t{64}, std::size_t{4096},
                             std::size_t{60000}}) {
        run(count, false);
        run(count, true);
    }
    return 0;
}
