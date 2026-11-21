cmake_minimum_required(VERSION 3.20)
project(LabNeura LANGUAGES CXX)

include(CheckCXXCompilerFlag)
include(CheckCXXSourceCompiles)
include(CTest)
include(FetchContent)
include(GNUInstallDirs)

option(
  LABNEURA_ENABLE_COVERAGE
  "Enable compiler-native source coverage instrumentation"
  OFF
)

option(
  LABNEURA_ENABLE_SANITIZERS
  "Enable AddressSanitizer and UndefinedBehaviorSanitizer"
  OFF
)

option(LABNEURA_BUILD_CORE "Build the domain-independent Core component" ON)
if(DEFINED LABNEURA_BUILD_ACCEL_CORE)
  set(LABNEURA_BUILD_CORE "${LABNEURA_BUILD_ACCEL_CORE}" CACHE BOOL
      "Build the domain-independent Core component" FORCE)
else()
  set(LABNEURA_BUILD_ACCEL_CORE "${LABNEURA_BUILD_CORE}" CACHE BOOL
      "Deprecated compatibility alias for LABNEURA_BUILD_CORE" FORCE)
endif()

option(
  LABNEURA_BUILD_TENSOR
  "Build the Tensor component"
  ON
)

option(
  LABNEURA_BUILD_EDA
  "Build the native EDA component"
  ON
)

option(
  LABNEURA_ENABLE_SLANG_VALIDATION
  "Enable externally provisioned Slang differential-validation targets"
  OFF
)

option(
  LABNEURA_BUILD_PYTHON_EDA_EXTENSION
  "Build the opt-in standalone _eda Python extension"
  OFF
)

option(
  LABNEURA_BUILD_PYTHON_TENSOR_EXTENSION
  "Build the independent _tensor Python extension"
  OFF
)

option(
  LABNEURA_BUILD_COMPAT
  "Build the monolithic labneura compatibility archive"
  ON
)

if(LABNEURA_BUILD_COMPAT AND
   (NOT LABNEURA_BUILD_ACCEL_CORE OR
    NOT LABNEURA_BUILD_TENSOR OR
    NOT LABNEURA_BUILD_EDA))
  message(FATAL_ERROR
    "LABNEURA_BUILD_COMPAT requires LABNEURA_BUILD_ACCEL_CORE, "
    "LABNEURA_BUILD_TENSOR, and LABNEURA_BUILD_EDA to be ON"
  )
endif()

if(LABNEURA_BUILD_TENSOR AND NOT LABNEURA_BUILD_ACCEL_CORE)
  message(FATAL_ERROR
    "LABNEURA_BUILD_TENSOR requires LABNEURA_BUILD_ACCEL_CORE=ON"
  )
endif()

if(LABNEURA_BUILD_PYTHON_TENSOR_EXTENSION AND NOT SKBUILD)
  message(FATAL_ERROR
    "LABNEURA_BUILD_PYTHON_TENSOR_EXTENSION requires SKBUILD=ON"
  )
endif()

if(LABNEURA_BUILD_PYTHON_TENSOR_EXTENSION AND NOT LABNEURA_BUILD_TENSOR)
  message(FATAL_ERROR
    "LABNEURA_BUILD_PYTHON_TENSOR_EXTENSION requires LABNEURA_BUILD_TENSOR=ON"
  )
endif()

if(LABNEURA_BUILD_PYTHON_TENSOR_EXTENSION AND LABNEURA_BUILD_COMPAT)
  message(FATAL_ERROR
    "LABNEURA_BUILD_PYTHON_TENSOR_EXTENSION requires "
    "LABNEURA_BUILD_COMPAT=OFF so the split and legacy modules stay separate"
  )
endif()

if(SKBUILD AND NOT LABNEURA_BUILD_COMPAT AND
   NOT LABNEURA_BUILD_PYTHON_TENSOR_EXTENSION AND
   NOT LABNEURA_BUILD_PYTHON_EDA_EXTENSION)
  message(FATAL_ERROR
    "SKBUILD requires LABNEURA_BUILD_COMPAT=ON unless an independent "
    "LABNEURA_BUILD_PYTHON_TENSOR_EXTENSION or "
    "LABNEURA_BUILD_PYTHON_EDA_EXTENSION is enabled"
  )
endif()

if(LABNEURA_BUILD_PYTHON_EDA_EXTENSION AND NOT SKBUILD)
  message(FATAL_ERROR
    "LABNEURA_BUILD_PYTHON_EDA_EXTENSION requires SKBUILD=ON"
  )
endif()

if(LABNEURA_BUILD_PYTHON_EDA_EXTENSION AND NOT LABNEURA_BUILD_EDA)
  message(FATAL_ERROR
    "LABNEURA_BUILD_PYTHON_EDA_EXTENSION requires LABNEURA_BUILD_EDA=ON"
  )
endif()


set(LABNEURA_SPLIT_PYTHON_DISTRIBUTION OFF)
if(SKBUILD AND NOT LABNEURA_BUILD_COMPAT AND
   (LABNEURA_BUILD_PYTHON_TENSOR_EXTENSION OR
    LABNEURA_BUILD_PYTHON_EDA_EXTENSION))
  set(LABNEURA_SPLIT_PYTHON_DISTRIBUTION ON)
endif()

# Optimized by default: slang is very slow in unoptimized builds.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Reuse compiled objects across rebuilds when ccache is installed.
find_program(LABNEURA_CCACHE ccache)
if(LABNEURA_CCACHE AND NOT DEFINED CMAKE_CXX_COMPILER_LAUNCHER)
  set(CMAKE_CXX_COMPILER_LAUNCHER "${LABNEURA_CCACHE}")
  message(STATUS "ccache enabled: ${LABNEURA_CCACHE}")
endif()

# Optional persistent dependency cache (skips re-cloning/rebuilding slang).
if(DEFINED ENV{LABNEURA_DEPS_DIR})
  set(FETCHCONTENT_BASE_DIR "$ENV{LABNEURA_DEPS_DIR}/${CMAKE_BUILD_TYPE}")
endif()

# ======================================
# C++ Standard
# ======================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
   AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 13)
  message(FATAL_ERROR
    "LabNeura requires GCC 13 or newer for source builds; detected "
    "GCC ${CMAKE_CXX_COMPILER_VERSION}. Install a prebuilt wheel with "
    "'python -m pip install labneura' or select a newer compiler using "
    "CC and CXX."
  )
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# CMake reports the build host in CMAKE_SYSTEM_PROCESSOR for a same-host macOS
# cross-architecture build. Normalize it to the explicitly requested single
# architecture so backend and test registration describe the produced binary.
if(APPLE AND CMAKE_OSX_ARCHITECTURES STREQUAL "x86_64")
  set(CMAKE_SYSTEM_PROCESSOR "x86_64")
elseif(APPLE AND CMAKE_OSX_ARCHITECTURES STREQUAL "arm64")
  set(CMAKE_SYSTEM_PROCESSOR "arm64")
endif()

if(LABNEURA_BUILD_EDA)
  set(LABNEURA_VERILOG_LEXER_TARGET_ARM64 OFF)
  set(LABNEURA_VERILOG_LEXER_TARGET_X86 OFF)
  if(APPLE AND CMAKE_OSX_ARCHITECTURES)
    if(CMAKE_OSX_ARCHITECTURES STREQUAL "arm64")
      set(LABNEURA_VERILOG_LEXER_TARGET_ARM64 ON)
    elseif(CMAKE_OSX_ARCHITECTURES STREQUAL "x86_64")
      set(LABNEURA_VERILOG_LEXER_TARGET_X86 ON)
    endif()
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64|ARM64|AARCH64)$")
    set(LABNEURA_VERILOG_LEXER_TARGET_ARM64 ON)
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|i.86)$")
    set(LABNEURA_VERILOG_LEXER_TARGET_X86 ON)
  endif()

  set(LABNEURA_VERILOG_LEXER_HAS_NEON OFF)
  if(LABNEURA_VERILOG_LEXER_TARGET_ARM64)
    check_cxx_source_compiles(
      "#include <arm_neon.h>
       int main() {
         const uint8x16_t value = vdupq_n_u8(1);
         return vgetq_lane_u8(value, 0) == 1 ? 0 : 1;
       }"
      LABNEURA_VERILOG_LEXER_COMPILER_HAS_NEON
    )
    if(LABNEURA_VERILOG_LEXER_COMPILER_HAS_NEON)
      set(LABNEURA_VERILOG_LEXER_HAS_NEON ON)
    endif()
  endif()

  set(LABNEURA_VERILOG_LEXER_HAS_AVX2 OFF)
  if(LABNEURA_VERILOG_LEXER_TARGET_X86)
    if(MSVC)
      check_cxx_compiler_flag("/arch:AVX2" LABNEURA_VERILOG_LEXER_COMPILER_HAS_AVX2)
      set(LABNEURA_VERILOG_LEXER_AVX2_FLAG "/arch:AVX2")
    else()
      check_cxx_compiler_flag("-mavx2" LABNEURA_VERILOG_LEXER_COMPILER_HAS_AVX2)
      set(LABNEURA_VERILOG_LEXER_AVX2_FLAG "-mavx2")
    endif()
    if(LABNEURA_VERILOG_LEXER_COMPILER_HAS_AVX2)
      set(LABNEURA_VERILOG_LEXER_HAS_AVX2 ON)
    endif()
  endif()

if(LABNEURA_BUILD_EDA AND LABNEURA_ENABLE_SLANG_VALIDATION)
  # Optional external differential-validation frontend.
  set(SLANG_INCLUDE_TESTS OFF CACHE BOOL "" FORCE)
  set(SLANG_INCLUDE_TOOLS OFF CACHE BOOL "" FORCE)
  # EDA's installed CMake target exports slang::slang. Keep Slang's C++
  # development install out of Python wheels, but ship it for C++ consumers.
  if(SKBUILD OR LABNEURA_SPLIT_PYTHON_DISTRIBUTION)
    set(SLANG_INCLUDE_INSTALL OFF CACHE BOOL "" FORCE)
  else()
    set(SLANG_INCLUDE_INSTALL ON CACHE BOOL "" FORCE)
  endif()
  set(SLANG_USE_MIMALLOC OFF CACHE BOOL "" FORCE)
  # Always use slang's pinned fmt; newer system fmt versions break its build.
  set(FETCHCONTENT_TRY_FIND_PACKAGE_MODE NEVER)
  set(SLANG_INCLUDE_PYLIB OFF CACHE BOOL "" FORCE)
  set(SLANG_INCLUDE_DOCS OFF CACHE BOOL "" FORCE)

  find_package(slang CONFIG REQUIRED)
endif()
endif()

# ======================================
# Architecture detection
# ======================================
if(APPLE AND LABNEURA_BUILD_TENSOR)
  if(CMAKE_OSX_ARCHITECTURES)
    set(ARCH "${CMAKE_OSX_ARCHITECTURES}")
  else()
    execute_process(
      COMMAND uname -m
      OUTPUT_VARIABLE ARCH
      OUTPUT_STRIP_TRAILING_WHITESPACE)
  endif()

  if(ARCH MATCHES "arm64|aarch64")
    message(STATUS "Apple Silicon detected (arm64) – NEON enabled")
  endif()
endif()

# ======================================
# Component object libraries and archives
# ======================================
set(LABNEURA_OBJECT_TARGETS "")

include(cmake/components/core/CMakeLists.txt)
include(cmake/components/tensor/CMakeLists.txt)
include(cmake/components/eda/CMakeLists.txt)

if(LABNEURA_BUILD_ACCEL_CORE)
  labneura_add_core_component()
endif()

if(LABNEURA_BUILD_TENSOR)
  labneura_add_tensor_component()
endif()

if(LABNEURA_BUILD_EDA)
  labneura_add_eda_component()
  add_library(labneura_eda_frontend ALIAS labneura_eda_objects)
  add_library(labneura_eda_lowering ALIAS labneura_eda_objects)
endif()

if(LABNEURA_BUILD_COMPAT)
  add_library(labneura STATIC
    $<TARGET_OBJECTS:labneura_core_objects>
    $<TARGET_OBJECTS:labneura_tensor_objects>
    $<TARGET_OBJECTS:labneura_eda_objects>
  )
  set_target_properties(labneura PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    EXPORT_NAME LabNeura
  )
  target_include_directories(labneura
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      ${CMAKE_CURRENT_BINARY_DIR}/generated
  )
  target_compile_features(labneura PUBLIC cxx_std_20)
endif()

if(LABNEURA_ENABLE_COVERAGE OR "$ENV{LABNEURA_COVERAGE}" STREQUAL "1")
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(LABNEURA_COVERAGE_COMPILE_OPTIONS
      -fprofile-instr-generate -fcoverage-mapping)
    set(LABNEURA_COVERAGE_LINK_OPTIONS -fprofile-instr-generate)
    set(LABNEURA_COVERAGE_FORMAT "llvm")
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(LABNEURA_COVERAGE_COMPILE_OPTIONS --coverage -O0)
    set(LABNEURA_COVERAGE_LINK_OPTIONS --coverage)
    set(LABNEURA_COVERAGE_FORMAT "gcov")
  else()
    message(FATAL_ERROR
      "LABNEURA_ENABLE_COVERAGE supports Clang/AppleClang or GCC; use the "
      "platform coverage workflow for MSVC")
  endif()
  foreach(target IN LISTS LABNEURA_OBJECT_TARGETS)
    target_compile_options(${target} PRIVATE ${LABNEURA_COVERAGE_COMPILE_OPTIONS})
  endforeach()
  foreach(target labneura_core labneura_tensor labneura_eda labneura)
    if(TARGET ${target})
      target_link_options(${target} INTERFACE ${LABNEURA_COVERAGE_LINK_OPTIONS})
    endif()
  endforeach()
  message(STATUS "LabNeura: ${LABNEURA_COVERAGE_FORMAT} coverage instrumentation enabled")
endif()

if(LABNEURA_ENABLE_SANITIZERS)
  if(MSVC)
    message(FATAL_ERROR "LABNEURA_ENABLE_SANITIZERS currently requires Clang or GCC")
  endif()
  foreach(target IN LISTS LABNEURA_OBJECT_TARGETS)
    target_compile_options(${target} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
  endforeach()
  foreach(target labneura_core labneura_tensor labneura_eda labneura)
    if(TARGET ${target})
      target_link_options(${target} INTERFACE -fsanitize=address,undefined)
    endif()
  endforeach()
endif()

# ======================================
# Compiler optimizations and warnings
# ======================================
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  # Base optimizations common to Clang and GCC
  foreach(target IN LISTS LABNEURA_OBJECT_TARGETS)
    target_compile_options(${target} PRIVATE
      -Wall
      -Wextra
      -Wpedantic
      -O3
      -ffast-math
      -funroll-loops
      -fomit-frame-pointer
      -fno-math-errno
    )
  endforeach()

  # Enable vectorization flags per compiler
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # GCC: tree vectorizer is enabled at -O3, but -ftree-vectorize is accepted
    foreach(target IN LISTS LABNEURA_OBJECT_TARGETS)
      target_compile_options(${target} PRIVATE -ftree-vectorize)
    endforeach()
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # Clang: auto-vectorization enabled at -O3; avoid non-portable -fvectorize/-fslp-vectorize
    # No extra flags needed
  endif()

  if(APPLE AND (CMAKE_OSX_ARCHITECTURES STREQUAL "arm64"
                OR (NOT CMAKE_OSX_ARCHITECTURES
                    AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")))
    # Apple M1/M2 optimizations (APPLE-only)
    foreach(target IN LISTS LABNEURA_OBJECT_TARGETS)
      target_compile_options(${target} PRIVATE
        -mcpu=apple-m1
        -mtune=native
      )
    endforeach()
    message(STATUS "M1 optimization flags enabled: -mcpu=apple-m1 -mtune=native")
  elseif(NOT APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    # Generic ARM64 optimizations for Linux/other platforms
    foreach(target IN LISTS LABNEURA_OBJECT_TARGETS)
      target_compile_options(${target} PRIVATE -mtune=generic)
    endforeach()
    message(STATUS "ARM64 optimization flags enabled: -mtune=generic")
  elseif((APPLE AND CMAKE_OSX_ARCHITECTURES STREQUAL "x86_64")
         OR CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    # Intel/AMD optimizations
    # Note: Using -mtune=generic for PyPI wheel compatibility (not -march=native)
    # Runtime CPU detection selects optimal backend (AVX2, SSE4.1, or Generic)
    foreach(target IN LISTS LABNEURA_OBJECT_TARGETS)
      target_compile_options(${target} PRIVATE -mtune=generic)
    endforeach()
    message(STATUS "x86_64 optimization flags enabled (portable): -mtune=generic")
  endif()
endif()

# ======================================
# Example executable
# ======================================
if(NOT SKBUILD AND LABNEURA_BUILD_COMPAT)
  add_executable(labneura_example examples/main.cpp)
  target_link_libraries(labneura_example PRIVATE labneura)
endif()

if(NOT SKBUILD AND LABNEURA_BUILD_TENSOR AND BUILD_TESTING)
    add_executable(labneura_backend_selection_test
      tests/backend_selection_test.cpp
    )
    target_include_directories(labneura_backend_selection_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src/labneura/tensor/core/dispatch
      ${CMAKE_CURRENT_SOURCE_DIR}/src/labneura/core
    )
    find_package(Threads REQUIRED)
    target_link_libraries(labneura_backend_selection_test PRIVATE labneura_tensor Threads::Threads)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64"
       AND LABNEURA_COMPILER_SUPPORTS_AVX2)
      target_compile_definitions(labneura_backend_selection_test
        PRIVATE LABNEURA_TEST_AVX2=1
      )
    endif()
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
      target_compile_definitions(labneura_backend_selection_test
        PRIVATE LABNEURA_TEST_NEON=1
      )
    endif()
    add_test(
      NAME backend_selection
      COMMAND labneura_backend_selection_test
    )

    add_executable(labneura_thread_config_test
      tests/thread_config_test.cpp
    )
    target_include_directories(labneura_thread_config_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src/labneura/tensor/core/runtime
    )
    target_include_directories(labneura_thread_config_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src/labneura
    )
    target_link_libraries(labneura_thread_config_test PRIVATE
      labneura_tensor Threads::Threads
    )
    add_test(NAME thread_configuration COMMAND labneura_thread_config_test)

    add_executable(labneura_parallel_for_test
      tests/parallel_for_test.cpp
    )
    target_link_libraries(labneura_parallel_for_test PRIVATE
      labneura_tensor Threads::Threads
    )
    add_test(NAME parallel_for_execution COMMAND labneura_parallel_for_test)

    add_executable(labneura_fp32_parallel_add_test
      tests/fp32_parallel_add_test.cpp
    )
    target_include_directories(labneura_fp32_parallel_add_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/tests
    )
    target_link_libraries(labneura_fp32_parallel_add_test PRIVATE labneura_tensor)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
      target_compile_definitions(labneura_fp32_parallel_add_test
        PRIVATE LABNEURA_TEST_NEON=1
      )
    endif()
    add_test(NAME fp32_parallel_add COMMAND labneura_fp32_parallel_add_test)

    add_executable(labneura_unsupported_dtype_test
      tests/unsupported_tensor_dtype_test.cpp
    )
    target_include_directories(labneura_unsupported_dtype_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/tests
    )
    target_link_libraries(labneura_unsupported_dtype_test PRIVATE labneura_tensor)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64"
       AND LABNEURA_COMPILER_SUPPORTS_AVX2)
      target_compile_definitions(labneura_unsupported_dtype_test
        PRIVATE LABNEURA_TEST_AVX2=1
      )
    endif()
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
      target_compile_definitions(labneura_unsupported_dtype_test
        PRIVATE LABNEURA_TEST_NEON=1
      )
    endif()
    add_test(
      NAME unsupported_tensor_dtypes
      COMMAND labneura_unsupported_dtype_test
    )

    add_executable(labneura_simd_int8_regression_test
      tests/simd_int8_regression_test.cpp
    )
    target_include_directories(labneura_simd_int8_regression_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/tests
    )
    target_link_libraries(labneura_simd_int8_regression_test PRIVATE labneura_tensor)
    add_executable(labneura_native_backend_benchmark
      tests/native_backend_benchmark.cpp
    )
    target_link_libraries(labneura_native_backend_benchmark PRIVATE labneura_tensor)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64"
       AND LABNEURA_COMPILER_SUPPORTS_AVX2)
      target_compile_definitions(labneura_simd_int8_regression_test
        PRIVATE LABNEURA_TEST_AVX2=1
      )
      target_compile_definitions(labneura_native_backend_benchmark
        PRIVATE LABNEURA_TEST_AVX2=1
      )
    endif()
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
      target_compile_definitions(labneura_simd_int8_regression_test
        PRIVATE LABNEURA_TEST_NEON=1
      )
      target_compile_definitions(labneura_native_backend_benchmark
        PRIVATE LABNEURA_TEST_NEON=1
      )
    endif()
    add_test(
      NAME avx2_int8_multiplication
      COMMAND labneura_simd_int8_regression_test AVX2
    )
    add_test(
      NAME neon_int8_subtraction
      COMMAND labneura_simd_int8_regression_test NEON
    )
    set_tests_properties(
      avx2_int8_multiplication
      neon_int8_subtraction
      PROPERTIES SKIP_RETURN_CODE 77
    )

    add_executable(labneura_avx2_integer_arithmetic_test
      tests/avx2_integer_arithmetic_test.cpp
    )
    target_include_directories(labneura_avx2_integer_arithmetic_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/tests
    )
    target_link_libraries(labneura_avx2_integer_arithmetic_test PRIVATE labneura_tensor)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64"
       AND LABNEURA_COMPILER_SUPPORTS_AVX2)
      target_compile_definitions(labneura_avx2_integer_arithmetic_test
        PRIVATE LABNEURA_TEST_AVX2=1
      )
    endif()
    add_test(
      NAME avx2_integer_arithmetic
      COMMAND labneura_avx2_integer_arithmetic_test
    )
    set_tests_properties(
      avx2_integer_arithmetic
      PROPERTIES SKIP_RETURN_CODE 77
    )

    add_executable(labneura_fp16_regression_test
      tests/fp16_regression_test.cpp
    )
    target_include_directories(labneura_fp16_regression_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/tests
      ${CMAKE_CURRENT_SOURCE_DIR}/src/labneura/tensor/core/dispatch
    )
    target_link_libraries(labneura_fp16_regression_test PRIVATE labneura_tensor)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64"
       AND LABNEURA_COMPILER_SUPPORTS_AVX2)
      target_compile_definitions(labneura_fp16_regression_test
        PRIVATE LABNEURA_TEST_AVX2=1
      )
    endif()
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
      target_compile_definitions(labneura_fp16_regression_test
        PRIVATE LABNEURA_TEST_NEON=1
      )
    endif()
    add_executable(labneura_fp32_direct_output_test
      tests/fp32_direct_output_test.cpp
    )
    target_include_directories(labneura_fp32_direct_output_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/tests
    )
    target_link_libraries(labneura_fp32_direct_output_test PRIVATE labneura_tensor)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64"
       AND LABNEURA_COMPILER_SUPPORTS_AVX2)
      target_compile_definitions(labneura_fp32_direct_output_test
        PRIVATE LABNEURA_TEST_AVX2=1
      )
    endif()
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
      target_compile_definitions(labneura_fp32_direct_output_test
        PRIVATE LABNEURA_TEST_NEON=1
      )
    endif()
    add_test(NAME fp32_generic_allocating COMMAND labneura_fp32_direct_output_test GENERIC)
    add_test(NAME fp32_avx2_allocating COMMAND labneura_fp32_direct_output_test AVX2)
    add_test(NAME fp32_neon_allocating COMMAND labneura_fp32_direct_output_test NEON)
    set_tests_properties(
      fp32_avx2_allocating
      fp32_neon_allocating
      PROPERTIES SKIP_RETURN_CODE 77
    )

    add_executable(labneura_add_into_test
      tests/add_into_test.cpp
    )
    target_include_directories(labneura_add_into_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/tests
    )
    target_link_libraries(labneura_add_into_test PRIVATE labneura_tensor)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64"
       AND LABNEURA_COMPILER_SUPPORTS_AVX2)
      target_compile_definitions(labneura_add_into_test
        PRIVATE LABNEURA_TEST_AVX2=1
      )
    endif()
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
      target_compile_definitions(labneura_add_into_test
        PRIVATE LABNEURA_TEST_NEON=1
      )
    endif()
    add_test(NAME add_into_generic COMMAND labneura_add_into_test GENERIC)
    add_test(NAME add_into_avx2 COMMAND labneura_add_into_test AVX2)
    add_test(NAME add_into_neon COMMAND labneura_add_into_test NEON)
    set_tests_properties(
      add_into_avx2
      add_into_neon
      PROPERTIES SKIP_RETURN_CODE 77
    )

    add_executable(labneura_tensor_nd_test
      tests/tensor_nd_test.cpp
    )
    target_include_directories(labneura_tensor_nd_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/tests
      ${CMAKE_CURRENT_SOURCE_DIR}/src/labneura/tensor/core/dispatch
    )
    target_link_libraries(labneura_tensor_nd_test PRIVATE labneura_tensor)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64"
       AND LABNEURA_COMPILER_SUPPORTS_AVX2)
      target_compile_definitions(labneura_tensor_nd_test
        PRIVATE LABNEURA_TEST_AVX2=1
      )
    endif()
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
      target_compile_definitions(labneura_tensor_nd_test
        PRIVATE LABNEURA_TEST_NEON=1
      )
    endif()
    add_test(NAME tensor_nd_generic COMMAND labneura_tensor_nd_test GENERIC)
    add_test(NAME tensor_nd_avx2 COMMAND labneura_tensor_nd_test AVX2)
    add_test(NAME tensor_nd_neon COMMAND labneura_tensor_nd_test NEON)
    set_tests_properties(
      tensor_nd_avx2
      tensor_nd_neon
      PROPERTIES SKIP_RETURN_CODE 77
    )

    add_executable(labneura_tensor_broadcast_test
      tests/tensor_broadcast_test.cpp
    )
    target_include_directories(labneura_tensor_broadcast_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/tests
    )
    target_link_libraries(labneura_tensor_broadcast_test PRIVATE labneura_tensor)
    add_test(NAME tensor_broadcast_generic
      COMMAND labneura_tensor_broadcast_test
    )
    add_test(NAME tensor_broadcast_fp16_generic
      COMMAND labneura_tensor_broadcast_test FP16
    )

    add_executable(labneura_tensor_comparison_test
      tests/tensor_comparison_test.cpp
    )
    target_include_directories(labneura_tensor_comparison_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/tests
    )
    target_link_libraries(labneura_tensor_comparison_test PRIVATE labneura_tensor)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64"
       AND LABNEURA_COMPILER_SUPPORTS_AVX2)
      target_compile_definitions(labneura_tensor_comparison_test
        PRIVATE LABNEURA_TEST_AVX2=1
      )
    endif()
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
      target_compile_definitions(labneura_tensor_comparison_test
        PRIVATE LABNEURA_TEST_NEON=1
      )
    endif()
    add_test(NAME tensor_comparison_generic
      COMMAND labneura_tensor_comparison_test GENERIC
    )
    add_test(NAME tensor_comparison_neon
      COMMAND labneura_tensor_comparison_test NEON
    )
    add_test(NAME tensor_comparison_avx2
      COMMAND labneura_tensor_comparison_test AVX2
    )
    set_tests_properties(
      tensor_comparison_neon
      tensor_comparison_avx2
      PROPERTIES SKIP_RETURN_CODE 77
    )

    add_executable(labneura_bool_dtype_test
      tests/bool_dtype_test.cpp
    )
    target_include_directories(labneura_bool_dtype_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/tests
    )
    target_link_libraries(labneura_bool_dtype_test PRIVATE labneura_tensor)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64"
       AND LABNEURA_COMPILER_SUPPORTS_AVX2)
      target_compile_definitions(labneura_bool_dtype_test
        PRIVATE LABNEURA_TEST_AVX2=1
      )
    endif()
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
      target_compile_definitions(labneura_bool_dtype_test
        PRIVATE LABNEURA_TEST_NEON=1
      )
    endif()
    add_test(NAME bool_dtype_generic
      COMMAND labneura_bool_dtype_test GENERIC
    )
    add_test(NAME bool_dtype_neon
      COMMAND labneura_bool_dtype_test NEON
    )
    add_test(NAME bool_dtype_avx2
      COMMAND labneura_bool_dtype_test AVX2
    )
    set_tests_properties(
      bool_dtype_neon
      bool_dtype_avx2
      PROPERTIES SKIP_RETURN_CODE 77
    )

    add_executable(labneura_aligned_allocation_safety_test
      tests/aligned_allocation_safety_test.cpp
    )
    target_link_libraries(labneura_aligned_allocation_safety_test
      PRIVATE labneura_tensor
    )
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64"
       AND LABNEURA_COMPILER_SUPPORTS_AVX2)
      target_compile_definitions(labneura_aligned_allocation_safety_test
        PRIVATE LABNEURA_TEST_AVX2=1
      )
    endif()
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
      target_compile_definitions(labneura_aligned_allocation_safety_test
        PRIVATE LABNEURA_TEST_NEON=1
      )
    endif()
    add_test(
      NAME aligned_allocation_safety
      COMMAND labneura_aligned_allocation_safety_test UTILITY
    )
    add_test(
      NAME avx2_allocation_safety
      COMMAND labneura_aligned_allocation_safety_test AVX2
    )
    set_tests_properties(
      avx2_allocation_safety
      PROPERTIES SKIP_RETURN_CODE 77
    )

    add_test(NAME fp16_f16c_dispatch COMMAND labneura_fp16_regression_test DISPATCH)
    add_test(NAME avx2_fp16_arithmetic COMMAND labneura_fp16_regression_test AVX2)
    add_test(NAME neon_fp16_arithmetic COMMAND labneura_fp16_regression_test NEON)
    set_tests_properties(
      avx2_fp16_arithmetic
      neon_fp16_arithmetic
      PROPERTIES SKIP_RETURN_CODE 77
    )
endif()

if(LABNEURA_BUILD_EDA AND NOT SKBUILD)
  if(BUILD_TESTING)
    add_executable(labneura_eda_verilog_lexer_test
      tests/eda_verilog_lexer_test.cpp
    )
    target_compile_definitions(labneura_eda_verilog_lexer_test PRIVATE
      LABNEURA_LEXER_CORPUS_PATH="${CMAKE_CURRENT_SOURCE_DIR}/tests/data/verilog/lexer_corpus.sv"
    )
    target_link_libraries(labneura_eda_verilog_lexer_test PRIVATE
      labneura_eda
    )
    if(LABNEURA_ENABLE_SLANG_VALIDATION)
      target_compile_definitions(labneura_eda_verilog_lexer_test PRIVATE
        LABNEURA_ENABLE_SLANG_LEXER_ORACLE=1
      )
      target_link_libraries(labneura_eda_verilog_lexer_test PRIVATE
        slang::slang
      )
    endif()
    target_include_directories(labneura_eda_verilog_lexer_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    add_test(NAME eda_verilog_lexer COMMAND labneura_eda_verilog_lexer_test)
    add_test(
      NAME eda_verilog_lexer_scalar
      COMMAND labneura_eda_verilog_lexer_test SCALAR
    )
    if(LABNEURA_VERILOG_LEXER_HAS_NEON)
      add_test(
        NAME eda_verilog_lexer_neon
        COMMAND labneura_eda_verilog_lexer_test NEON
      )
      set_tests_properties(
        eda_verilog_lexer_neon
        PROPERTIES SKIP_RETURN_CODE 77
      )
    endif()
    add_test(
      NAME eda_verilog_lexer_avx2
      COMMAND labneura_eda_verilog_lexer_test AVX2
    )
    set_tests_properties(
      eda_verilog_lexer_avx2
      PROPERTIES SKIP_RETURN_CODE 77
    )

  endif()

  if(BUILD_TESTING AND LABNEURA_ENABLE_SLANG_VALIDATION)

    add_executable(labneura_eda_verilog_preprocessor_test
      tests/eda_verilog_preprocessor_test.cpp
    )
    target_link_libraries(labneura_eda_verilog_preprocessor_test PRIVATE
      labneura_eda
      slang::slang
    )
    add_test(
      NAME eda_verilog_preprocessor
      COMMAND labneura_eda_verilog_preprocessor_test
    )

    add_executable(labneura_eda_verilog_syntax_test
      tests/eda_verilog_syntax_test.cpp
    )
    target_compile_definitions(labneura_eda_verilog_syntax_test PRIVATE
      LABNEURA_SYNTAX_FIXTURE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/tests/data/verilog/parser_foundation.sv"
      LABNEURA_BEHAVIORAL_FIXTURE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/tests/data/verilog/behavioral_rtl.sv"
    )
    target_link_libraries(labneura_eda_verilog_syntax_test PRIVATE
      labneura_eda
      slang::slang
    )
    add_test(
      NAME eda_verilog_syntax
      COMMAND labneura_eda_verilog_syntax_test
    )

    add_executable(labneura_eda_verilog_semantic_test
      tests/eda_verilog_semantic_test.cpp
    )
    target_link_libraries(labneura_eda_verilog_semantic_test PRIVATE
      labneura_eda
    )
    add_test(
      NAME eda_verilog_semantic
      COMMAND labneura_eda_verilog_semantic_test
    )

    add_executable(labneura_eda_verilog_elaboration_test
      tests/eda_verilog_elaboration_test.cpp
    )
    target_link_libraries(labneura_eda_verilog_elaboration_test PRIVATE
      labneura_eda
      slang::slang
    )
    add_test(
      NAME eda_verilog_elaboration
      COMMAND labneura_eda_verilog_elaboration_test
    )

    add_executable(labneura_eda_verilog_type_test
      tests/eda_verilog_type_test.cpp
    )
    target_link_libraries(labneura_eda_verilog_type_test PRIVATE
      labneura_eda
      slang::slang
    )
    add_test(
      NAME eda_verilog_type
      COMMAND labneura_eda_verilog_type_test
    )

    add_executable(labneura_eda_verilog_callable_test
      tests/eda_verilog_callable_test.cpp
    )
    target_link_libraries(labneura_eda_verilog_callable_test PRIVATE
      labneura_eda
      slang::slang
    )
    add_test(
      NAME eda_verilog_callable
      COMMAND labneura_eda_verilog_callable_test
    )

    add_executable(labneura_eda_verilog_legacy_ports_test
      tests/eda_verilog_legacy_ports_test.cpp
    )
    target_link_libraries(labneura_eda_verilog_legacy_ports_test PRIVATE
      labneura_eda
      slang::slang
    )
    add_test(
      NAME eda_verilog_legacy_ports
      COMMAND labneura_eda_verilog_legacy_ports_test
    )

    add_executable(labneura_eda_verilog_callable_binding_test
      tests/eda_verilog_callable_binding_test.cpp
    )
    target_link_libraries(labneura_eda_verilog_callable_binding_test PRIVATE
      labneura_eda
    )
    add_test(
      NAME eda_verilog_callable_binding
      COMMAND labneura_eda_verilog_callable_binding_test
    )

    add_executable(labneura_eda_verilog_lvalue_test
      tests/eda_verilog_lvalue_test.cpp
    )
    target_link_libraries(labneura_eda_verilog_lvalue_test PRIVATE labneura_eda)
    add_test(NAME eda_verilog_lvalue COMMAND labneura_eda_verilog_lvalue_test)

    add_executable(labneura_eda_verilog_assignment_test
      tests/eda_verilog_assignment_test.cpp
    )
    target_link_libraries(labneura_eda_verilog_assignment_test PRIVATE labneura_eda)
    add_test(NAME eda_verilog_assignment COMMAND labneura_eda_verilog_assignment_test)

    add_executable(labneura_eda_verilog_function_evaluation_test
      tests/eda_verilog_function_evaluation_test.cpp
    )
    target_link_libraries(labneura_eda_verilog_function_evaluation_test PRIVATE
      labneura_eda
    )
    add_test(
      NAME eda_verilog_function_evaluation
      COMMAND labneura_eda_verilog_function_evaluation_test
    )

    add_executable(labneura_eda_verilog_task_execution_test
      tests/eda_verilog_task_execution_test.cpp
    )
    target_link_libraries(labneura_eda_verilog_task_execution_test PRIVATE
      labneura_eda
    )
    add_test(
      NAME eda_verilog_task_execution
      COMMAND labneura_eda_verilog_task_execution_test
    )

    add_executable(labneura_eda_verilog_control_flow_test
      tests/eda_verilog_control_flow_test.cpp
    )
    target_link_libraries(labneura_eda_verilog_control_flow_test PRIVATE labneura_eda)
    add_test(NAME eda_verilog_control_flow
      COMMAND labneura_eda_verilog_control_flow_test)

    add_executable(labneura_eda_verilog_loop_execution_test
      tests/eda_verilog_loop_execution_test.cpp
    )
    target_link_libraries(labneura_eda_verilog_loop_execution_test PRIVATE labneura_eda)
    add_test(NAME eda_verilog_loop_execution
      COMMAND labneura_eda_verilog_loop_execution_test)

    add_executable(labneura_eda_verilog_branch_execution_test
      tests/eda_verilog_branch_execution_test.cpp
    )
    target_link_libraries(labneura_eda_verilog_branch_execution_test PRIVATE labneura_eda)
    add_test(NAME eda_verilog_branch_execution
      COMMAND labneura_eda_verilog_branch_execution_test)

    add_executable(labneura_eda_verilog_delay_execution_test
      tests/eda_verilog_delay_execution_test.cpp
    )
    target_link_libraries(labneura_eda_verilog_delay_execution_test PRIVATE
      labneura_eda
    )
    add_test(
      NAME eda_verilog_delay_execution
      COMMAND labneura_eda_verilog_delay_execution_test
    )

    add_executable(labneura_eda_verilog_event_execution_test
      tests/eda_verilog_event_execution_test.cpp
    )
    target_link_libraries(labneura_eda_verilog_event_execution_test PRIVATE
      labneura_eda
    )
    add_test(
      NAME eda_verilog_event_execution
      COMMAND labneura_eda_verilog_event_execution_test
    )

    add_executable(labneura_eda_verilog_process_analysis_test
      tests/eda_verilog_process_analysis_test.cpp
    )
    target_link_libraries(labneura_eda_verilog_process_analysis_test PRIVATE labneura_eda)
    add_test(NAME eda_verilog_process_analysis
      COMMAND labneura_eda_verilog_process_analysis_test)

    add_executable(labneura_eda_verilog_process_dependencies_test
      tests/eda_verilog_process_dependencies_test.cpp
    )
    target_link_libraries(labneura_eda_verilog_process_dependencies_test PRIVATE labneura_eda)
    add_test(NAME eda_verilog_process_dependencies
      COMMAND labneura_eda_verilog_process_dependencies_test)

    add_executable(labneura_eda_verilog_assignment_coverage_test
      tests/eda_verilog_assignment_coverage_test.cpp
    )
    target_link_libraries(labneura_eda_verilog_assignment_coverage_test PRIVATE labneura_eda)
    add_test(NAME eda_verilog_assignment_coverage
      COMMAND labneura_eda_verilog_assignment_coverage_test)

    add_executable(labneura_eda_verilog_sequential_analysis_test
      tests/eda_verilog_sequential_analysis_test.cpp
    )
    target_link_libraries(labneura_eda_verilog_sequential_analysis_test PRIVATE labneura_eda)
    add_test(NAME eda_verilog_sequential_analysis
      COMMAND labneura_eda_verilog_sequential_analysis_test)

    add_executable(labneura_eda_verilog_synthesis_eligibility_test
      tests/eda_verilog_synthesis_eligibility_test.cpp
    )
    target_link_libraries(labneura_eda_verilog_synthesis_eligibility_test PRIVATE labneura_eda)
    add_test(NAME eda_verilog_synthesis_eligibility
      COMMAND labneura_eda_verilog_synthesis_eligibility_test)

    add_library(labneura_eda_netlist_test_support STATIC
      tests/support/eda_netlist_evaluator.cpp
      tests/support/eda_sequential_evaluator.cpp)
    target_include_directories(labneura_eda_netlist_test_support PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    target_link_libraries(labneura_eda_netlist_test_support PUBLIC labneura_eda)

    add_executable(labneura_eda_netlist_evaluator_test
      tests/eda_netlist_evaluator_test.cpp)
    target_link_libraries(labneura_eda_netlist_evaluator_test PRIVATE
      labneura_eda_netlist_test_support)
    add_test(NAME eda_netlist_evaluator
      COMMAND labneura_eda_netlist_evaluator_test)

    add_executable(labneura_eda_verilog_combinational_equivalence_test
      tests/eda_verilog_combinational_equivalence_test.cpp)
    target_compile_definitions(labneura_eda_verilog_combinational_equivalence_test
      PRIVATE
        "LABNEURA_NATIVE_COMB_FIXTURE_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/tests/data/verilog/native_equivalence/comb\"")
    target_link_libraries(labneura_eda_verilog_combinational_equivalence_test
      PRIVATE labneura_eda_netlist_test_support)
    add_test(NAME eda_verilog_combinational_equivalence
      COMMAND labneura_eda_verilog_combinational_equivalence_test)

    add_executable(labneura_eda_verilog_sequential_equivalence_test
      tests/eda_verilog_sequential_equivalence_test.cpp)
    target_link_libraries(labneura_eda_verilog_sequential_equivalence_test
      PRIVATE labneura_eda_netlist_test_support)
    add_test(NAME eda_verilog_sequential_equivalence
      COMMAND labneura_eda_verilog_sequential_equivalence_test)

    add_executable(labneura_eda_verilog_randomized_lowering_test
      tests/eda_verilog_randomized_lowering_test.cpp)
    target_link_libraries(labneura_eda_verilog_randomized_lowering_test PRIVATE
      labneura_eda_netlist_test_support)
    add_test(NAME eda_verilog_randomized_lowering
      COMMAND labneura_eda_verilog_randomized_lowering_test)

    set_tests_properties(
      eda_netlist_evaluator
      eda_verilog_combinational_equivalence
      eda_verilog_sequential_equivalence
      PROPERTIES TIMEOUT 120)

    foreach(lowering_test IN ITEMS expression combinational sequential module)
      add_executable(labneura_eda_verilog_${lowering_test}_lowering_test
        tests/eda_verilog_${lowering_test}_lowering_test.cpp)
      target_link_libraries(labneura_eda_verilog_${lowering_test}_lowering_test
        PRIVATE labneura_eda)
      add_test(NAME eda_verilog_${lowering_test}_lowering
        COMMAND labneura_eda_verilog_${lowering_test}_lowering_test)
    endforeach()

    add_executable(labneura_eda_native_verilog_project_context_test
      tests/eda_native_verilog_project_context_test.cpp)
    target_include_directories(labneura_eda_native_verilog_project_context_test
      PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(labneura_eda_native_verilog_project_context_test
      PRIVATE labneura_eda)
    add_test(NAME eda_native_verilog_project_context
      COMMAND labneura_eda_native_verilog_project_context_test)

    add_executable(labneura_eda_native_verilog_project_projection_test
      tests/eda_native_verilog_project_projection_test.cpp)
    target_include_directories(labneura_eda_native_verilog_project_projection_test
      PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(labneura_eda_native_verilog_project_projection_test
      PRIVATE labneura_eda)
    add_test(NAME eda_native_verilog_project_projection
      COMMAND labneura_eda_native_verilog_project_projection_test)

    add_executable(labneura_eda_verilog_native_lowering_differential_test
      tests/eda_verilog_native_lowering_differential_test.cpp)
    target_link_libraries(labneura_eda_verilog_native_lowering_differential_test
      PRIVATE labneura_eda slang::slang)
    add_test(NAME eda_verilog_native_lowering_differential
      COMMAND labneura_eda_verilog_native_lowering_differential_test)

    add_executable(labneura_eda_verilog_process_differential_test
      tests/eda_verilog_process_differential_test.cpp
    )
    target_link_libraries(labneura_eda_verilog_process_differential_test PRIVATE
      labneura_eda
      slang::slang
    )
    add_test(NAME eda_verilog_process_differential
      COMMAND labneura_eda_verilog_process_differential_test)

    add_executable(labneura_eda_verilog_callable_differential_test
      tests/eda_verilog_callable_differential_test.cpp
    )
    target_link_libraries(labneura_eda_verilog_callable_differential_test PRIVATE
      labneura_eda
      slang::slang
    )
    add_test(
      NAME eda_verilog_callable_differential
      COMMAND labneura_eda_verilog_callable_differential_test
    )

    add_executable(labneura_eda_verilog_assignment_differential_test
      tests/eda_verilog_assignment_differential_test.cpp
    )
    target_link_libraries(labneura_eda_verilog_assignment_differential_test PRIVATE
      labneura_eda
      slang::slang
    )
    add_test(
      NAME eda_verilog_assignment_differential
      COMMAND labneura_eda_verilog_assignment_differential_test
    )

    add_executable(labneura_eda_verilog_control_flow_differential_test
      tests/eda_verilog_control_flow_differential_test.cpp
    )
    target_link_libraries(labneura_eda_verilog_control_flow_differential_test PRIVATE
      labneura_eda
      slang::slang
    )
    add_test(
      NAME eda_verilog_control_flow_differential
      COMMAND labneura_eda_verilog_control_flow_differential_test
    )
  endif()

  # The formal-equivalence harness invokes the external Yosys executable via a
  # POSIX shell. The dedicated Linux Yosys CI job owns that integration test;
  # keep Windows EDA-only CTest free of an unavoidable external-tool skip.
  if(BUILD_TESTING AND NOT WIN32)
    add_executable(labneura_eda_verilog_native_formal_equivalence_test
      tests/eda_verilog_native_formal_equivalence_test.cpp)
    target_compile_definitions(labneura_eda_verilog_native_formal_equivalence_test
      PRIVATE
        "LABNEURA_NATIVE_FORMAL_FIXTURE_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/tests/data/verilog/native_equivalence/formal\"")
    target_link_libraries(labneura_eda_verilog_native_formal_equivalence_test
      PRIVATE labneura_eda)
    add_test(NAME eda_verilog_native_formal_equivalence
      COMMAND labneura_eda_verilog_native_formal_equivalence_test)
    set_tests_properties(eda_verilog_native_formal_equivalence PROPERTIES
      SKIP_RETURN_CODE 77 TIMEOUT 90)
  endif()

  add_executable(labneura_lexer_benchmark EXCLUDE_FROM_ALL
    benchmarks/eda/verilog_lexer_benchmark.cpp
  )
  target_compile_definitions(labneura_lexer_benchmark PRIVATE
    LABNEURA_LEXER_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
  )
  target_link_libraries(labneura_lexer_benchmark PRIVATE labneura_eda)
  target_include_directories(labneura_lexer_benchmark PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
  )

  add_executable(labneura_preprocessor_benchmark EXCLUDE_FROM_ALL
    benchmarks/eda/verilog_preprocessor_benchmark.cpp
  )
  target_compile_definitions(labneura_preprocessor_benchmark PRIVATE
    LABNEURA_LEXER_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
  )
  target_link_libraries(labneura_preprocessor_benchmark PRIVATE labneura_eda)

  add_executable(labneura_native_frontend_profile EXCLUDE_FROM_ALL
    benchmarks/eda/native_frontend_profile.cpp
  )
  target_compile_definitions(labneura_native_frontend_profile PRIVATE
    LABNEURA_FRONTEND_PROFILE_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
  )
  target_link_libraries(labneura_native_frontend_profile PRIVATE labneura_eda)

  if(LABNEURA_ENABLE_SLANG_VALIDATION)
    add_executable(labneura_lexer_slang_benchmark EXCLUDE_FROM_ALL
      benchmarks/eda/verilog_lexer_benchmark.cpp
    )
    target_compile_definitions(labneura_lexer_slang_benchmark PRIVATE
      LABNEURA_LEXER_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
      LABNEURA_ENABLE_SLANG_LEXER_BENCHMARK=1
    )
    target_link_libraries(labneura_lexer_slang_benchmark PRIVATE
      labneura_eda
      slang::slang
    )
    target_include_directories(labneura_lexer_slang_benchmark PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )

    add_executable(labneura_slang_frontend_profile EXCLUDE_FROM_ALL
      benchmarks/eda/slang_frontend_profile.cpp
    )
    target_link_libraries(labneura_slang_frontend_profile PRIVATE
      labneura_eda
      slang::slang
    )
  endif()

  add_executable(labneura_semantic_reference_index_benchmark EXCLUDE_FROM_ALL
    benchmarks/eda/semantic_reference_index_benchmark.cpp
  )
  target_link_libraries(labneura_semantic_reference_index_benchmark PRIVATE
    labneura_eda)

  if(BUILD_TESTING)
    add_executable(labneura_eda_frontend_profile_schema_test
      tests/eda_frontend_profile_schema_test.cpp
    )
    target_include_directories(labneura_eda_frontend_profile_schema_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/eda
    )
    add_test(NAME eda_frontend_profile_schema
      COMMAND labneura_eda_frontend_profile_schema_test)

    add_executable(labneura_eda_semantic_reference_index_experiment_test
      tests/eda_semantic_reference_index_experiment_test.cpp)
    target_link_libraries(labneura_eda_semantic_reference_index_experiment_test
      PRIVATE labneura_eda)
    add_test(NAME eda_semantic_reference_index_experiment
      COMMAND labneura_eda_semantic_reference_index_experiment_test)

    add_executable(labneura_eda_semantic_reference_index_test
      tests/eda_semantic_reference_index_test.cpp)
    target_link_libraries(labneura_eda_semantic_reference_index_test
      PRIVATE labneura_eda)
    add_test(NAME eda_semantic_reference_index
      COMMAND labneura_eda_semantic_reference_index_test)
  endif()
endif()

if(LABNEURA_BUILD_EDA AND NOT SKBUILD AND BUILD_TESTING)
  add_executable(labneura_eda_verilog_native_lowering_test
    tests/eda_verilog_native_lowering_test.cpp
    tests/support/eda_netlist_evaluator.cpp)
  target_include_directories(labneura_eda_verilog_native_lowering_test
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  target_link_libraries(labneura_eda_verilog_native_lowering_test
    PRIVATE labneura_eda)
  add_test(NAME eda_verilog_native_lowering
    COMMAND labneura_eda_verilog_native_lowering_test)
endif()

if(LABNEURA_BUILD_EDA AND NOT SKBUILD AND BUILD_TESTING AND
   NOT LABNEURA_ENABLE_SLANG_VALIDATION)
  add_executable(labneura_eda_native_verilog_project_context_test
    tests/eda_native_verilog_project_context_test.cpp)
  target_include_directories(labneura_eda_native_verilog_project_context_test
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(labneura_eda_native_verilog_project_context_test
    PRIVATE labneura_eda)
  add_test(NAME eda_native_verilog_project_context
    COMMAND labneura_eda_native_verilog_project_context_test)

  add_executable(labneura_eda_native_verilog_project_projection_test
    tests/eda_native_verilog_project_projection_test.cpp)
  target_include_directories(labneura_eda_native_verilog_project_projection_test
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(labneura_eda_native_verilog_project_projection_test
    PRIVATE labneura_eda)
  add_test(NAME eda_native_verilog_project_projection
    COMMAND labneura_eda_native_verilog_project_projection_test)
endif()

if(SKBUILD)
  set(Python_FIND_FRAMEWORK FIRST)
  if(WIN32)
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
    set(LABNEURA_PYTHON_EXTENSION_SUFFIX ".pyd")
  else()
    find_package(Python COMPONENTS Interpreter REQUIRED)
    execute_process(
      COMMAND "${Python_EXECUTABLE}" -c
        "import pathlib, sys, sysconfig; version = sysconfig.get_config_var('VERSION'); candidates = [sysconfig.get_path('include'), sysconfig.get_config_var('INCLUDEPY') or '', str(pathlib.Path(sysconfig.get_path('stdlib')).parents[1] / 'include' / ('python' + version)), str(pathlib.Path(sys.prefix) / 'include' / ('python' + version))]; print(next((path for path in candidates if path and (pathlib.Path(path) / 'Python.h').is_file()), ''))"
      OUTPUT_VARIABLE LABNEURA_PYTHON_INCLUDE_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE LABNEURA_PYTHON_INCLUDE_RESULT
    )
    if(NOT LABNEURA_PYTHON_INCLUDE_RESULT EQUAL 0 OR
       NOT EXISTS "${LABNEURA_PYTHON_INCLUDE_DIR}/Python.h")
      message(FATAL_ERROR "Could not locate Python.h for ${Python_EXECUTABLE}")
    endif()
    add_library(Python::Module INTERFACE IMPORTED)
    target_include_directories(Python::Module INTERFACE
      "${LABNEURA_PYTHON_INCLUDE_DIR}"
    )
    set(LABNEURA_PYTHON_EXTENSION_SUFFIX "${CMAKE_SHARED_MODULE_SUFFIX}")
  endif()
  find_package(pybind11 CONFIG REQUIRED)

  if(LABNEURA_BUILD_COMPAT)
    add_library(labneura_python MODULE
      python/labneura_py.cpp
      python/tensor_bindings.cpp
      python/eda_bindings.cpp
    )
    set_target_properties(labneura_python PROPERTIES
      PREFIX ""
      OUTPUT_NAME labneura
      SUFFIX "${LABNEURA_PYTHON_EXTENSION_SUFFIX}"
    )
    target_link_libraries(labneura_python PRIVATE pybind11::module labneura)
    if(LABNEURA_ENABLE_COVERAGE OR "$ENV{LABNEURA_COVERAGE}" STREQUAL "1")
      target_compile_options(labneura_python PRIVATE ${LABNEURA_COVERAGE_COMPILE_OPTIONS})
      target_link_options(labneura_python PRIVATE ${LABNEURA_COVERAGE_LINK_OPTIONS})
    endif()
    if(APPLE)
      target_link_options(labneura_python PRIVATE
        "-undefined" "dynamic_lookup"
      )
    endif()
    install(TARGETS labneura_python
      LIBRARY DESTINATION .
      RUNTIME DESTINATION .
    )
  endif()

  if(LABNEURA_BUILD_PYTHON_TENSOR_EXTENSION)
    add_library(_tensor MODULE
      python/tensor_module.cpp
      python/tensor_bindings.cpp
    )
    set_target_properties(_tensor PROPERTIES
      PREFIX ""
      OUTPUT_NAME _tensor
      SUFFIX "${LABNEURA_PYTHON_EXTENSION_SUFFIX}"
      POSITION_INDEPENDENT_CODE ON
    )
    target_link_libraries(_tensor PRIVATE pybind11::module labneura_tensor)
    if(LABNEURA_ENABLE_COVERAGE OR "$ENV{LABNEURA_COVERAGE}" STREQUAL "1")
      target_compile_options(_tensor PRIVATE ${LABNEURA_COVERAGE_COMPILE_OPTIONS})
      target_link_options(_tensor PRIVATE ${LABNEURA_COVERAGE_LINK_OPTIONS})
    endif()
    if(APPLE)
      target_link_options(_tensor PRIVATE
        "-undefined" "dynamic_lookup"
      )
    endif()
    install(TARGETS _tensor
      LIBRARY DESTINATION .
      RUNTIME DESTINATION .
    )
  endif()

  if(LABNEURA_BUILD_PYTHON_EDA_EXTENSION)
    add_library(_eda MODULE
      python/eda_module.cpp
      python/eda_bindings.cpp
    )
    set_target_properties(_eda PROPERTIES
      PREFIX ""
      OUTPUT_NAME _eda
      SUFFIX "${LABNEURA_PYTHON_EXTENSION_SUFFIX}"
      POSITION_INDEPENDENT_CODE ON
    )
    target_link_libraries(_eda PRIVATE pybind11::module labneura_eda)
    if(LABNEURA_ENABLE_COVERAGE OR "$ENV{LABNEURA_COVERAGE}" STREQUAL "1")
      target_compile_options(_eda PRIVATE ${LABNEURA_COVERAGE_COMPILE_OPTIONS})
      target_link_options(_eda PRIVATE ${LABNEURA_COVERAGE_LINK_OPTIONS})
    endif()
    if(APPLE)
      target_link_options(_eda PRIVATE
        "-undefined" "dynamic_lookup"
      )
    endif()
    install(TARGETS _eda
      LIBRARY DESTINATION .
      RUNTIME DESTINATION .
    )

    if(TARGET labneura_python)
      target_compile_definitions(labneura_python PRIVATE
        LABNEURA_USE_STANDALONE_EDA_MODULE=1
      )
      add_dependencies(labneura_python _eda)
    endif()
  endif()
endif()

# ======================================
# Install rules
# ======================================
set(LABNEURA_INSTALL_TARGETS "")
if(TARGET labneura_core)
  list(APPEND LABNEURA_INSTALL_TARGETS labneura_core)
endif()
if(TARGET labneura_tensor)
  list(APPEND LABNEURA_INSTALL_TARGETS labneura_tensor)
endif()
if(TARGET labneura_eda)
  list(APPEND LABNEURA_INSTALL_TARGETS labneura_eda)
endif()
if(TARGET labneura)
  list(APPEND LABNEURA_INSTALL_TARGETS labneura)
endif()
include(CMakePackageConfigHelpers)

if(LABNEURA_INSTALL_TARGETS AND NOT SKBUILD AND NOT LABNEURA_SPLIT_PYTHON_DISTRIBUTION)
  install(TARGETS ${LABNEURA_INSTALL_TARGETS}
    EXPORT LabNeuraTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )

  if(TARGET labneura)
    install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  else()
    if(TARGET labneura_core)
      install(DIRECTORY include/labneura/core/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/labneura/core)
      install(FILES include/labneura/backends/common/cpu_features.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/labneura/backends/common)
    endif()
    if(TARGET labneura_tensor)
      install(DIRECTORY include/labneura/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/labneura
        PATTERN "eda" EXCLUDE
        PATTERN "cpu_features_test.h" EXCLUDE)
    endif()
    if(TARGET labneura_eda)
      install(DIRECTORY include/labneura/eda/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/labneura/eda)
    endif()
  endif()

  install(EXPORT LabNeuraTargets
    FILE LabNeuraTargets.cmake
    NAMESPACE LabNeura::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/LabNeura
  )

  set(LABNEURA_PACKAGE_NEEDS_SLANG OFF)
  if(TARGET labneura_eda)
    set(LABNEURA_PACKAGE_NEEDS_SLANG ON)
  endif()
  configure_package_config_file(
    cmake/LabNeuraConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/LabNeuraConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/LabNeura
  )
  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/LabNeuraConfigVersion.cmake"
    VERSION 0.5.2
    COMPATIBILITY SameMajorVersion
  )
  install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/LabNeuraConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/LabNeuraConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/LabNeura
  )
endif()
