cmake_minimum_required(VERSION 3.14)
project(LabNeura LANGUAGES CXX)

include(CheckCXXCompilerFlag)
# ======================================
# C++ Standard
# ======================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ======================================
# Architecture detection
# ======================================
if(APPLE)
  execute_process(
    COMMAND uname -m
    OUTPUT_VARIABLE ARCH
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  if(ARCH MATCHES "arm64|aarch64")
    add_compile_definitions(LABNEURA_HAVE_M1=1)
    message(STATUS "Apple Silicon detected (arm64) – NEON enabled")
  endif()
endif()

# ======================================
# Core library
# ======================================
add_library(labneura
  src/labneura/tensor.cpp
  src/labneura/backends/base.cpp
  src/labneura/backends/generic.cpp
  src/labneura/backends/neon.cpp
  src/labneura/backends/avx2.cpp
  src/labneura/backends/sse41.cpp
  src/labneura/backends/cpu_features.cpp
  src/labneura/backends/backend_factory.cpp
)

if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  # Check for AVX2 support
  check_cxx_compile_flag("-mavx2", LABNEURA_COMPILER_SUPPORTS_AVX2)
  if(LABNEURA_COMPILER_SUPPORTS_AVX2)
    set_source_files_properties(src/labneura/backends/avx2.cpp PROPERTIES COMPILE_FLAGS "-mavx2")
    target_compile_definitions(labneura PRIVATE LABNEURA_ENABLE_AVX2=1)
    message(STATUS "LabNeura: AVX2 backend enabled (-mavx2)")
  else()
    message(STATUS "LabNeura: AVX2 backend disabled (compiler lacks -mavx2)")
  endif()

  # Check for SSE4.1 support (fallback for older Intel/AMD CPUs)
  check_cxx_compile_flag("-msse4.1", LABNEURA_COMPILER_SUPPORTS_SSE41)
  if(LABNEURA_COMPILER_SUPPORTS_SSE41)
    set_source_files_properties(src/labneura/backends/sse41.cpp PROPERTIES COMPILE_FLAGS "-msse4.1")
    target_compile_definitions(labneura PRIVATE LABNEURA_ENABLE_SSE41=1)
    message(STATUS "LabNeura: SSE4.1 backend enabled (-msse4.1)")
  else()
    message(STATUS "LabNeura: SSE4.1 backend disabled (compiler lacks -msse4.1)")
  endif()
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
  target_compile_definitions(labneura PRIVATE LABNEURA_ENABLE_NEON=1)
endif()

target_include_directories(labneura
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# ======================================
# Compiler optimizations and warnings
# ======================================
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  # Base optimizations common to Clang and GCC
  target_compile_options(labneura PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -O3
    -ffast-math
    -funroll-loops
    -fomit-frame-pointer
    -fno-math-errno
  )

  # Enable vectorization flags per compiler
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # GCC: tree vectorizer is enabled at -O3, but -ftree-vectorize is accepted
    target_compile_options(labneura PRIVATE -ftree-vectorize)
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # Clang: auto-vectorization enabled at -O3; avoid non-portable -fvectorize/-fslp-vectorize
    # No extra flags needed
  endif()

  if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64" AND APPLE)
    # Apple M1/M2 optimizations (APPLE-only)
    target_compile_options(labneura PRIVATE
      -mcpu=apple-m1
      -mtune=native
    )
    message(STATUS "M1 optimization flags enabled: -mcpu=apple-m1 -mtune=native")
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    # Generic ARM64 optimizations for Linux/other platforms
    target_compile_options(labneura PRIVATE
      -mtune=generic
    )
    message(STATUS "ARM64 optimization flags enabled: -mtune=generic")
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    # Intel/AMD optimizations
    # Note: Using -mtune=generic for PyPI wheel compatibility (not -march=native)
    # Runtime CPU detection selects optimal backend (AVX2, SSE4.1, or Generic)
    target_compile_options(labneura PRIVATE
      -mtune=generic
    )
    message(STATUS "x86_64 optimization flags enabled (portable): -mtune=generic")
  endif()
endif()

# ======================================
# Example executable
# ======================================
add_executable(labneura_example examples/main.cpp)
target_link_libraries(labneura_example PRIVATE labneura)

# ======================================
# Install rules
# ======================================
install(TARGETS labneura EXPORT LabNeuraTargets)
install(DIRECTORY include/ DESTINATION include)

# ======================================
# Package versioning
# ======================================
include(CMakePackageConfigHelpers)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/LabNeuraConfigVersion.cmake"
  VERSION 0.1
  COMPATIBILITY AnyNewerVersion
)