#include "labneura/core/cpu_features.h"

int main() {
    const auto features = labneura::detect_cpu_features();
    if (features.avx2 != labneura::cpu_supports_avx2()) return 1;
    if (features.f16c != labneura::cpu_supports_f16c()) return 2;
    if (labneura::cpu_supports_neon() && labneura::cpu_supports_avx2()) return 3;
    return 0;
}
