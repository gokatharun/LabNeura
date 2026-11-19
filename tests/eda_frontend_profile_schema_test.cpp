#include "frontend_profile_common.h"

#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {
using namespace labneura::eda::benchmark;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
} // namespace

int main() {
    try {
        std::size_t index = 0;
        const auto timing = measure(2, 3, [&] {
            if (++index > 5) throw std::runtime_error("unexpected pass count");
        });
        require(index == 5, "measurement did not run warmups plus repetitions");
        require(timing.minimum_seconds >= 0.0 &&
                    timing.median_seconds >= timing.minimum_seconds &&
                    timing.p95_seconds >= timing.median_seconds,
                "timing summary ordering is invalid");
        require(fnv1a64("abc") == "e71fa2190541574b",
                "content identity changed unexpectedly");
        require(json_escape("a\n\"b") == "a\\n\\\"b",
                "JSON escaping is invalid");

        std::ostringstream output;
        auto* previous = std::cout.rdbuf(output.rdbuf());
        emit_result("syntax", "native", "scalar", "fixture", "hash", 12, 3,
                    1, 1, 2, std::nullopt, std::nullopt, std::nullopt,
                    2, 3, timing,
                    "accepted", "Release");
        std::cout.rdbuf(previous);
        const auto json = output.str();
        require(json.find("\"peak_rss_bytes\":null") != std::string::npos &&
                    json.find("\"net_count\":null") != std::string::npos &&
                    json.find("\"minimum_seconds\":") != std::string::npos,
                "schema did not preserve unavailable metrics or timing fields");
        std::cout << "frontend profiling schema tests passed\n";
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
