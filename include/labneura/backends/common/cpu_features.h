#pragma once

namespace labneura {
    
struct CPUFeatures {
    bool sse41 = false;
    bool avx2 = false;
};

CPUFeatures detect_cpu_features();
bool cpu_supports_avx2();
bool cpu_supports_neon();
bool cpu_supports_sse41();

} // namespace labneura