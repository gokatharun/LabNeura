cmake_minimum_required(VERSION 3.20)
project(LabNeura LANGUAGES CXX)

include(CheckCXXCompilerFlag)
include(CTest)
include(FetchContent)
include(GNUInstallDirs)

option(
  LABNEURA_ENABLE_COVERAGE
  "Enable LLVM source-based coverage instrumentation"
  OFF
)

option(
  LABNEURA_BUILD_ACCEL_CORE
  "Build the application-independent acceleration core"
  ON
)

option(
  LABNEURA_BUILD_TENSOR
  "Build the Tensor component"
  ON
)

option(
  LABNEURA_BUILD_EDA
  "Build the EDA component and its Slang dependency"
  ON
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
if(LABNEURA_CCACHE)
  set(CMAKE_CXX_COMPILER_LAUNCHER "${LABNEURA_CCACHE}")
  message(STATUS "ccache enabled: ${LABNEURA_CCACHE}")
endif()

# Optional persistent dependency cache (skips re-cloning/rebuilding slang).
if(DEFINED ENV{LABNEURA_DEPS_DIR})
  set(FETCHCONTENT_BASE_DIR "$ENV{LABNEURA_DEPS_DIR}/${CMAKE_BUILD_TYPE}")
endif()

if(LABNEURA_BUILD_EDA)
  # Embed the SKY130 Liberty file so the default parser works in source
  # checkouts and installed wheels alike.
  set(LABNEURA_EDA_DEFAULT_LIBERTY
    "${CMAKE_SOURCE_DIR}/assets/sky130_fd_sc_hd__tt_025C_1v80.lib"
  )
  file(READ "${LABNEURA_EDA_DEFAULT_LIBERTY}" LABNEURA_EDA_LIBERTY_SOURCE)
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${LABNEURA_EDA_DEFAULT_LIBERTY}"
    "${CMAKE_SOURCE_DIR}/src/labneura/eda/parser/liberty/default_cell_library.h.in"
  )

  # Keep each C++ string literal below the minimum supported compiler limit.
  string(LENGTH "${LABNEURA_EDA_LIBERTY_SOURCE}" LABNEURA_EDA_LIBERTY_LENGTH)
  set(LABNEURA_EDA_LIBERTY_CHUNKS "")
  set(LABNEURA_EDA_LIBERTY_OFFSET 0)
  set(LABNEURA_EDA_LIBERTY_CHUNK_SIZE 60000)
  while(LABNEURA_EDA_LIBERTY_OFFSET LESS LABNEURA_EDA_LIBERTY_LENGTH)
    math(EXPR LABNEURA_EDA_LIBERTY_REMAINING
      "${LABNEURA_EDA_LIBERTY_LENGTH} - ${LABNEURA_EDA_LIBERTY_OFFSET}"
    )
    if(LABNEURA_EDA_LIBERTY_REMAINING LESS LABNEURA_EDA_LIBERTY_CHUNK_SIZE)
      set(LABNEURA_EDA_LIBERTY_CHUNK_SIZE
        "${LABNEURA_EDA_LIBERTY_REMAINING}"
      )
    endif()
    string(SUBSTRING "${LABNEURA_EDA_LIBERTY_SOURCE}"
      "${LABNEURA_EDA_LIBERTY_OFFSET}"
      "${LABNEURA_EDA_LIBERTY_CHUNK_SIZE}"
      LABNEURA_EDA_LIBERTY_CHUNK
    )
    string(APPEND LABNEURA_EDA_LIBERTY_CHUNKS
      "    R\"LABNEURA_CHUNK(${LABNEURA_EDA_LIBERTY_CHUNK})LABNEURA_CHUNK\",\n"
    )
    math(EXPR LABNEURA_EDA_LIBERTY_OFFSET
      "${LABNEURA_EDA_LIBERTY_OFFSET} + ${LABNEURA_EDA_LIBERTY_CHUNK_SIZE}"
    )
    set(LABNEURA_EDA_LIBERTY_CHUNK_SIZE 60000)
  endwhile()

  file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated/labneura/eda")
  configure_file(
    "${CMAKE_SOURCE_DIR}/src/labneura/eda/parser/liberty/default_cell_library.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/generated/labneura/eda/default_cell_library.h"
    @ONLY
  )
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

if(LABNEURA_BUILD_EDA)
  # ======================================
  # Verilog frontend
  # ======================================
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

  FetchContent_Declare(
    slang
    GIT_REPOSITORY https://github.com/MikePopoloski/slang.git
    GIT_TAG 7ddf4059f79eff508dd486eb42fd650cdf320d52
    GIT_PROGRESS TRUE
  )
  FetchContent_MakeAvailable(slang)
endif()

# ======================================
# Architecture detection
# ======================================
if(APPLE AND LABNEURA_BUILD_TENSOR)
  execute_process(
    COMMAND uname -m
    OUTPUT_VARIABLE ARCH
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  if(ARCH MATCHES "arm64|aarch64")
    message(STATUS "Apple Silicon detected (arm64) – NEON enabled")
  endif()
endif()

# ======================================
# Component object libraries and archives
# ======================================
set(LABNEURA_OBJECT_TARGETS "")

if(LABNEURA_BUILD_ACCEL_CORE)
  add_library(labneura_accel_core_objects OBJECT
    src/labneura/backends/common/cpu_features.cpp
  )
  set_target_properties(labneura_accel_core_objects PROPERTIES
    POSITION_INDEPENDENT_CODE ON
  )
  target_include_directories(labneura_accel_core_objects
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
  )

  add_library(labneura_accel_core STATIC
    $<TARGET_OBJECTS:labneura_accel_core_objects>
  )
  set_target_properties(labneura_accel_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    EXPORT_NAME AccelCore
  )
  target_include_directories(labneura_accel_core
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>
  )
  target_compile_features(labneura_accel_core PUBLIC cxx_std_20)
  list(APPEND LABNEURA_OBJECT_TARGETS labneura_accel_core_objects)
endif()

if(LABNEURA_BUILD_TENSOR)
  add_library(labneura_tensor_objects OBJECT
    src/labneura/tensor.cpp
    src/labneura/runtime.cpp
    src/labneura/parallel.cpp
    src/labneura/backends/common/base.cpp
    src/labneura/backends/common/tensor_storage.cpp
    src/labneura/backends/common/backend_factory.cpp
    src/labneura/backends/cpu/generic/generic_backend.cpp
    src/labneura/backends/cpu/generic/generic_ops_fp32.cpp
    src/labneura/backends/cpu/generic/generic_ops_compare.cpp
    src/labneura/backends/cpu/generic/generic_ops_logical.cpp
    src/labneura/backends/cpu/generic/generic_ops_int8.cpp
    src/labneura/backends/cpu/neon/neon_backend.cpp
    src/labneura/backends/cpu/neon/neon_ops_compare.cpp
    src/labneura/backends/cpu/neon/neon_ops_logical.cpp
    src/labneura/backends/cpu/neon/neon_ops_fp32.cpp
    src/labneura/backends/cpu/neon/neon_ops_int8.cpp
    src/labneura/backends/cpu/neon/neon_ops_int16.cpp
    src/labneura/backends/cpu/avx2/avx2_backend.cpp
    src/labneura/backends/cpu/avx2/avx2_ops_compare.cpp
    src/labneura/backends/cpu/avx2/avx2_ops_logical.cpp
    src/labneura/backends/cpu/avx2/avx2_ops_fp32.cpp
    src/labneura/backends/cpu/avx2/avx2_ops_int8.cpp
    src/labneura/backends/cpu/avx2/avx2_ops_fp16.cpp
    src/labneura/backends/cpu/avx2/avx2_ops_int16.cpp
  )
  set_target_properties(labneura_tensor_objects PROPERTIES
    POSITION_INDEPENDENT_CODE ON
  )
  target_include_directories(labneura_tensor_objects
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
  )

  add_library(labneura_tensor STATIC
    $<TARGET_OBJECTS:labneura_tensor_objects>
  )
  set_target_properties(labneura_tensor PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    EXPORT_NAME Tensor
  )
  target_include_directories(labneura_tensor
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>
  )
  target_compile_features(labneura_tensor PUBLIC cxx_std_20)
  if(TARGET labneura_accel_core)
    target_link_libraries(labneura_tensor PUBLIC labneura_accel_core)
  endif()
  list(APPEND LABNEURA_OBJECT_TARGETS labneura_tensor_objects)
endif()

if(LABNEURA_BUILD_EDA)
  add_library(labneura_eda_objects OBJECT
    src/labneura/eda/parser/verilog_parser.cpp
    src/labneura/eda/parser/liberty_parser.cpp
    src/labneura/eda/netlist/netlist_json.cpp
    src/labneura/eda/netlist/netlist_verilog.cpp
    src/labneura/eda/netlist/netlist_graph.cpp
    src/labneura/eda/netlist/netlist_blif.cpp
    src/labneura/eda/mapping/generic/generic_mapping.cpp
    src/labneura/eda/mapping/generic/behavioral_synthesis.cpp
    src/labneura/eda/mapping/technology/technology_mapping.cpp
    src/labneura/eda/mapping/fpga/fpga_mapping.cpp
    src/labneura/eda/mapping/fpga/fpga_library.cpp
  )
  set_target_properties(labneura_eda_objects PROPERTIES
    POSITION_INDEPENDENT_CODE ON
  )
  target_include_directories(labneura_eda_objects
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      ${CMAKE_CURRENT_BINARY_DIR}/generated
  )
  target_link_libraries(labneura_eda_objects PRIVATE slang::slang)

  add_library(labneura_eda STATIC
    $<TARGET_OBJECTS:labneura_eda_objects>
  )
  set_target_properties(labneura_eda PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    EXPORT_NAME EDA
  )
  target_include_directories(labneura_eda
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>
  )
  target_compile_features(labneura_eda PUBLIC cxx_std_20)
  target_link_libraries(labneura_eda PRIVATE slang::slang)
  list(APPEND LABNEURA_OBJECT_TARGETS labneura_eda_objects)
endif()

if(LABNEURA_BUILD_COMPAT)
  add_library(labneura STATIC
    $<TARGET_OBJECTS:labneura_accel_core_objects>
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
  target_link_libraries(labneura PRIVATE slang::slang)
endif()

if(LABNEURA_ENABLE_COVERAGE OR "$ENV{LABNEURA_COVERAGE}" STREQUAL "1")
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR "LABNEURA_COVERAGE requires Clang")
  endif()
  foreach(target IN LISTS LABNEURA_OBJECT_TARGETS)
    target_compile_options(${target} PRIVATE
      -fprofile-instr-generate
      -fcoverage-mapping
    )
  endforeach()
  if(TARGET labneura)
    target_link_options(labneura INTERFACE -fprofile-instr-generate)
  endif()
  message(STATUS "LabNeura: LLVM coverage instrumentation enabled")
endif()

if(LABNEURA_BUILD_TENSOR AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  # Check for AVX2 support
  check_cxx_compiler_flag("-mavx2" LABNEURA_COMPILER_SUPPORTS_AVX2)
  if(LABNEURA_COMPILER_SUPPORTS_AVX2)
    # Apply AVX2 flags to all AVX2 backend files
    set_source_files_properties(
      src/labneura/backends/cpu/avx2/avx2_backend.cpp
      src/labneura/backends/cpu/avx2/avx2_ops_compare.cpp
      src/labneura/backends/cpu/avx2/avx2_ops_logical.cpp
      src/labneura/backends/cpu/avx2/avx2_ops_int8.cpp
      src/labneura/backends/cpu/avx2/avx2_ops_int16.cpp
      PROPERTIES COMPILE_FLAGS "-mavx2"
    )
    # FP16 and FP32 must keep IEEE NaN/Inf/division semantics, so opt out of
    # -ffast-math (Clang otherwise lowers vector division to RCPPS + Newton-Raphson).
    set_source_files_properties(
      src/labneura/backends/cpu/avx2/avx2_ops_fp32.cpp
      src/labneura/backends/cpu/avx2/avx2_ops_fp16.cpp
      src/labneura/backends/cpu/avx2/avx2_ops_compare.cpp
      PROPERTIES COMPILE_FLAGS "-mavx2 -fno-fast-math"
    )
    target_compile_definitions(labneura_tensor_objects PRIVATE LABNEURA_ENABLE_AVX2=1)
    message(STATUS "LabNeura: AVX2 backend enabled (-mavx2)")
    # F16C is independent of AVX2: build its kernel in an isolated translation
    # unit and select it at runtime only when the CPU reports support.
    check_cxx_compiler_flag("-mf16c" LABNEURA_COMPILER_SUPPORTS_F16C)
    if(LABNEURA_COMPILER_SUPPORTS_F16C)
      target_sources(labneura_tensor_objects PRIVATE
        src/labneura/backends/cpu/avx2/avx2_ops_fp16_f16c.cpp
      )
      set_source_files_properties(
        src/labneura/backends/cpu/avx2/avx2_ops_fp16_f16c.cpp
        PROPERTIES COMPILE_FLAGS "-mavx2 -mf16c -fno-fast-math"
      )
      target_compile_definitions(labneura_tensor_objects PRIVATE LABNEURA_ENABLE_AVX2_F16C=1)
      message(STATUS "LabNeura: AVX2 FP16 F16C kernel enabled (-mf16c, isolated)")
    endif()
  else()
    message(STATUS "LabNeura: AVX2 backend disabled (compiler lacks -mavx2)")
  endif()


endif()

if(LABNEURA_BUILD_TENSOR AND CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  # Generic is the numerical reference. Preserve IEEE behavior in floating
  # arithmetic and comparisons; fast-math assumes NaNs do not occur.
  set_source_files_properties(
    src/labneura/backends/cpu/generic/generic_ops_fp32.cpp
    src/labneura/backends/cpu/generic/generic_ops_compare.cpp
    PROPERTIES COMPILE_FLAGS "-fno-fast-math"
  )
endif()

if(LABNEURA_BUILD_TENSOR AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
  target_compile_definitions(labneura_tensor_objects PRIVATE LABNEURA_ENABLE_NEON=1)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    # FP32 NEON division is IEEE (vdivq_f32 / scalar fdiv); keep NaN/Inf/denormal
    # semantics independent of the global -ffast-math.
    set_source_files_properties(
      src/labneura/backends/cpu/neon/neon_ops_compare.cpp
      src/labneura/backends/cpu/neon/neon_ops_fp32.cpp
      PROPERTIES COMPILE_FLAGS "-fno-fast-math"
    )
  endif()
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

  if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64" AND APPLE)
    # Apple M1/M2 optimizations (APPLE-only)
    foreach(target IN LISTS LABNEURA_OBJECT_TARGETS)
      target_compile_options(${target} PRIVATE
        -mcpu=apple-m1
        -mtune=native
      )
    endforeach()
    message(STATUS "M1 optimization flags enabled: -mcpu=apple-m1 -mtune=native")
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    # Generic ARM64 optimizations for Linux/other platforms
    foreach(target IN LISTS LABNEURA_OBJECT_TARGETS)
      target_compile_options(${target} PRIVATE -mtune=generic)
    endforeach()
    message(STATUS "ARM64 optimization flags enabled: -mtune=generic")
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
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

  if(BUILD_TESTING)
    add_executable(labneura_backend_selection_test
      tests/backend_selection_test.cpp
    )
    target_include_directories(labneura_backend_selection_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src/labneura/backends/common
    )
    find_package(Threads REQUIRED)
    target_link_libraries(labneura_backend_selection_test PRIVATE labneura Threads::Threads)
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
      ${CMAKE_CURRENT_SOURCE_DIR}/src/labneura
    )
    target_link_libraries(labneura_thread_config_test PRIVATE
      labneura Threads::Threads
    )
    add_test(NAME thread_configuration COMMAND labneura_thread_config_test)

    add_executable(labneura_parallel_for_test
      tests/parallel_for_test.cpp
    )
    target_link_libraries(labneura_parallel_for_test PRIVATE
      labneura Threads::Threads
    )
    add_test(NAME parallel_for_execution COMMAND labneura_parallel_for_test)

    add_executable(labneura_fp32_parallel_add_test
      tests/fp32_parallel_add_test.cpp
    )
    target_link_libraries(labneura_fp32_parallel_add_test PRIVATE labneura)
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
    target_link_libraries(labneura_unsupported_dtype_test PRIVATE labneura)
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
    target_link_libraries(labneura_simd_int8_regression_test PRIVATE labneura)
    add_executable(labneura_native_backend_benchmark
      tests/native_backend_benchmark.cpp
    )
    target_link_libraries(labneura_native_backend_benchmark PRIVATE labneura)
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

    add_executable(labneura_fp16_regression_test
      tests/fp16_regression_test.cpp
    )
    target_include_directories(labneura_fp16_regression_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/tests
      ${CMAKE_CURRENT_SOURCE_DIR}/src/labneura/backends/common
    )
    target_link_libraries(labneura_fp16_regression_test PRIVATE labneura)
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
    target_link_libraries(labneura_fp32_direct_output_test PRIVATE labneura)
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
    target_link_libraries(labneura_add_into_test PRIVATE labneura)
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
      ${CMAKE_CURRENT_SOURCE_DIR}/src/labneura/backends/common
    )
    target_link_libraries(labneura_tensor_nd_test PRIVATE labneura)
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
    target_link_libraries(labneura_tensor_broadcast_test PRIVATE labneura)
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
    target_link_libraries(labneura_tensor_comparison_test PRIVATE labneura)
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
    target_link_libraries(labneura_bool_dtype_test PRIVATE labneura)
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
      PRIVATE labneura
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
endif()

if(SKBUILD)
  set(Python_FIND_FRAMEWORK FIRST)
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
      SUFFIX "${CMAKE_SHARED_MODULE_SUFFIX}"
    )
    target_link_libraries(labneura_python PRIVATE pybind11::module labneura)
    if(LABNEURA_ENABLE_COVERAGE OR "$ENV{LABNEURA_COVERAGE}" STREQUAL "1")
      target_compile_options(labneura_python PRIVATE
        -fprofile-instr-generate
        -fcoverage-mapping
      )
      target_link_options(labneura_python PRIVATE -fprofile-instr-generate)
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
      SUFFIX "${CMAKE_SHARED_MODULE_SUFFIX}"
      POSITION_INDEPENDENT_CODE ON
    )
    target_link_libraries(_tensor PRIVATE pybind11::module labneura_tensor)
    if(LABNEURA_ENABLE_COVERAGE OR "$ENV{LABNEURA_COVERAGE}" STREQUAL "1")
      target_compile_options(_tensor PRIVATE
        -fprofile-instr-generate
        -fcoverage-mapping
      )
      target_link_options(_tensor PRIVATE -fprofile-instr-generate)
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
      SUFFIX "${CMAKE_SHARED_MODULE_SUFFIX}"
      POSITION_INDEPENDENT_CODE ON
    )
    target_link_libraries(_eda PRIVATE pybind11::module labneura_eda)
    if(LABNEURA_ENABLE_COVERAGE OR "$ENV{LABNEURA_COVERAGE}" STREQUAL "1")
      target_compile_options(_eda PRIVATE
        -fprofile-instr-generate
        -fcoverage-mapping
      )
      target_link_options(_eda PRIVATE -fprofile-instr-generate)
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
if(TARGET labneura_accel_core)
  list(APPEND LABNEURA_INSTALL_TARGETS labneura_accel_core)
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
    if(TARGET labneura_accel_core)
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
    VERSION 0.4.2
    COMPATIBILITY SameMajorVersion
  )
  install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/LabNeuraConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/LabNeuraConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/LabNeura
  )
endif()
