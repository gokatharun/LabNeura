#!/usr/bin/env python3
"""
Split large backend files into modular components by data type.
"""

import os
import re

BACKENDS_DIR = "src/labneura/backends"

# Define split points for AVX2
AVX2_SPLITS = {
    "header": (1, 192),  # Includes, namespace, utilities, constructor, data access, clone
    "fp32": (193, 789),
    "int8": (790, 917),
    "fp16": (918, 1274),
    "int16": (1275, 1664),
    "footer": (1665, 1667)  # namespace close, endif
}

def read_lines(filepath, start, end):
    """Read specific line range from file (1-indexed, inclusive)."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
        return lines[start-1:end]

def write_file(filepath, content):
    """Write content to file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        f.writelines(content)
    print(f"✓ Created {filepath}")

def split_avx2():
    """Split AVX2 backend into modular files."""
    src_file = f"{BACKENDS_DIR}/avx2.cpp"
    
    # Read header and footer
    header = read_lines(src_file, *AVX2_SPLITS["header"])
    footer = read_lines(src_file, *AVX2_SPLITS["footer"])
    
    # Create internal header for shared utilities
    internal_h = [
        "#pragma once\n",
        "\n",
        "#if defined(__AVX2__)\n",
        "\n",
        "#include <immintrin.h>\n",
        "#include <algorithm>\n",
        "#include <cstdlib>\n",
        "\n",
        "#if defined(__GNUC__) || defined(__clang__)\n",
        "#include <cpuid.h>\n",
        "#endif\n",
        "\n",
        "namespace labneura {\n",
        "\n",
    ] + header[14:62] + [  # Just the utility functions
        "\n",
        "} // namespace labneura\n",
        "\n",
        "#endif // __AVX2__\n",
    ]
    write_file(f"{BACKENDS_DIR}/avx2/avx2_internal.h", internal_h)
    
    # Create main backend file (constructor, accessors, clone)
    backend_cpp = [
        "#if defined(__AVX2__)\n",
        "\n",
        "#include \"labneura/backends/avx2.h\"\n",
        "#include \"avx2_internal.h\"\n",
        "#include \"labneura/utils/alignment.hpp\"\n",
        "\n",
    ] + header[13:188] + footer  # Skip utilities, keep constructor & accessors
    write_file(f"{BACKENDS_DIR}/avx2/avx2_backend.cpp", backend_cpp)
    
    # Create FP32 ops file
    fp32_header = [
        "#if defined(__AVX2__)\n",
        "\n",
        "#include \"labneura/backends/avx2.h\"\n",
        "#include \"avx2_internal.h\"\n",
        "#include \"labneura/utils/alignment.hpp\"\n",
        "\n",
        "namespace labneura {\n",
        "\n",
    ]
    fp32_ops = read_lines(src_file, *AVX2_SPLITS["fp32"])
    write_file(f"{BACKENDS_DIR}/avx2/avx2_ops_fp32.cpp", fp32_header + fp32_ops + footer)
    
    # Create INT8 ops file
    int8_header = fp32_header.copy()
    int8_ops = read_lines(src_file, *AVX2_SPLITS["int8"])
    write_file(f"{BACKENDS_DIR}/avx2/avx2_ops_int8.cpp", int8_header + int8_ops + footer)
    
    # Create FP16 ops file
    fp16_header = fp32_header.copy()
    fp16_ops = read_lines(src_file, *AVX2_SPLITS["fp16"])
    write_file(f"{BACKENDS_DIR}/avx2/avx2_ops_fp16.cpp", fp16_header + fp16_ops + footer)
    
    # Create INT16 ops file
    int16_header = fp32_header.copy()
    int16_ops = read_lines(src_file, *AVX2_SPLITS["int16"])
    write_file(f"{BACKENDS_DIR}/avx2/avx2_ops_int16.cpp", int16_header + int16_ops + footer)

# Similar functions for Generic and NEON (simplified since they're smaller)
def split_generic():
    """Split Generic backend."""
    src_file = f"{BACKENDS_DIR}/generic.cpp"
    
    with open(src_file, 'r') as f:
        content = f.read()
    
    # Find the split points by searching for function signatures
    fp32_start = content.find("void GenericBackend::operation_fp32")
    int8_start = content.find("void GenericBackend::operation_int8")
    
    if fp32_start == -1 or int8_start == -1:
        print(f"✗ Could not find operation functions in {src_file}")
        return
    
    # Header content (everything before fp32 operations)
    header_content = content[:fp32_start]
    fp32_content = content[fp32_start:int8_start]
    int8_content = content[int8_start:]
    
    # Extract the namespace close and includes
    lines = content.split('\n')
    header_lines = [l for l in lines[:30] if l.strip()]  # First 30 lines usually have includes
    
    common_header = '''#include "labneura/backends/generic.h"
#include <algorithm>

namespace labneura {

'''
    
    footer = "\n} // namespace labneura\n"
    
    # Write internal header
    internal_h = '''#pragma once

namespace labneura {
// Add shared utilities here if needed
}
'''
    write_file(f"{BACKENDS_DIR}/generic/generic_internal.h", internal_h)
    
    # Write main backend file (constructor, accessors)
    backend_content = content[:fp32_start] + footer
    write_file(f"{BACKENDS_DIR}/generic/generic_backend.cpp", backend_content)
    
    # Write FP32 ops
    write_file(f"{BACKENDS_DIR}/generic/generic_ops_fp32.cpp", common_header + fp32_content + footer)
    
    # Write INT8 ops
    write_file(f"{BACKENDS_DIR}/generic/generic_ops_int8.cpp", common_header + int8_content)

def split_neon():
    """Split NEON backend - similar to AVX2 but for ARM."""
    src_file = f"{BACKENDS_DIR}/neon.cpp"
    
    with open(src_file, 'r') as f:
        lines = f.readlines()
    
    # Find operation functions
    op_indices = []
    for i, line in enumerate(lines, 1):
        if re.match(r'void NEONBackend::operation_(fp32|int8|int16)', line):
            op_indices.append(i)
    
    if len(op_indices) < 2:
        print(f"✗ Could not find enough operation functions in {src_file}")
        return
    
    # Approximate split (will need manual adjustment)
    header_end = op_indices[0] - 1
    fp32_end = op_indices[1] - 1 if len(op_indices) > 1 else len(lines)
    int8_end = op_indices[2] - 1 if len(op_indices) > 2 else len(lines)
    
    common_header = '''#if defined(__ARM_NEON) || defined(__aarch64__)

#include "labneura/backends/neon.h"
#include "neon_internal.h"
#include "labneura/utils/alignment.hpp"
#include <arm_neon.h>

namespace labneura {

'''
    
    footer = "\n} // namespace labneura\n\n#endif // __ARM_NEON || __aarch64__\n"
    
    # Write internal header
    internal_h = '''#pragma once

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>

namespace labneura {
// Add shared utilities here if needed
}

#endif
'''
    write_file(f"{BACKENDS_DIR}/neon/neon_internal.h", internal_h)
    
    # Write main backend
    backend_content = lines[:header_end]
    write_file(f"{BACKENDS_DIR}/neon/neon_backend.cpp", backend_content + [footer])
    
    # Write operation files
    fp32_lines = lines[op_indices[0]-1:fp32_end]
    write_file(f"{BACKENDS_DIR}/neon/neon_ops_fp32.cpp", [common_header] + fp32_lines + [footer])
    
    if len(op_indices) > 1:
        int8_lines = lines[op_indices[1]-1:int8_end]
        write_file(f"{BACKENDS_DIR}/neon/neon_ops_int8.cpp", [common_header] + int8_lines + [footer])
    
    if len(op_indices) > 2:
        int16_lines = lines[op_indices[2]-1:]
        write_file(f"{BACKENDS_DIR}/neon/neon_ops_int16.cpp", [common_header] + int16_lines)

if __name__ == "__main__":
    print("Starting backend modularization...\n")
    
    print("=== Splitting AVX2 Backend ===")
    split_avx2()
    
    print("\n=== Splitting Generic Backend ===")
    split_generic()
    
    print("\n=== Splitting NEON Backend ===")
    split_neon()
    
    print("\n✅ Modularization complete!")
    print("\nNext steps:")
    print("1. Update CMakeLists.txt")
    print("2. Update CMakeLists.txt and pyproject.toml")
    print("3. Verify build and tests")
