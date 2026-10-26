cmake_minimum_required(VERSION 3.20)
project(LabNeura LANGUAGES CXX)

include(CheckCXXCompilerFlag)
include(FetchContent)

# Embed the SKY130 Liberty file so the default parser works in source checkouts
# and installed wheels alike.
set(LABNEURA_EDA_DEFAULT_LIBERTY
  "${CMAKE_SOURCE_DIR}/assets/sky130_fd_sc_hd__tt_025C_1v80.lib"
)
file(READ "${LABNEURA_EDA_DEFAULT_LIBERTY}" LABNEURA_EDA_LIBERTY_SOURCE)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${LABNEURA_EDA_DEFAULT_LIBERTY}"
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
  "${CMAKE_SOURCE_DIR}/src/labneura/eda/default_cell_library.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/generated/labneura/eda/default_cell_library.h"
  @ONLY
)

# ======================================
# C++ Standard
# ======================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ======================================
# Verilog frontend
# ======================================
set(SLANG_INCLUDE_TESTS OFF CACHE BOOL "" FORCE)
set(SLANG_INCLUDE_TOOLS OFF CACHE BOOL "" FORCE)
set(SLANG_INCLUDE_INSTALL OFF CACHE BOOL "" FORCE)
set(SLANG_USE_MIMALLOC OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
  slang
  GIT_REPOSITORY https://github.com/MikePopoloski/slang.git
  GIT_TAG 7ddf4059f79eff508dd486eb42fd650cdf320d52
  GIT_PROGRESS TRUE
)
FetchContent_MakeAvailable(slang)

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
  src/labneura/eda/verilog_parser.cpp
  src/labneura/eda/netlist_json.cpp
  src/labneura/eda/cell_library.cpp
  src/labneura/tensor.cpp
  src/labneura/backends/common/base.cpp
  src/labneura/backends/common/cpu_features.cpp
  src/labneura/backends/common/backend_factory.cpp
  # CPU Generic backend (modular)
  src/labneura/backends/cpu/generic/generic_backend.cpp
  src/labneura/backends/cpu/generic/generic_ops_fp32.cpp
  src/labneura/backends/cpu/generic/generic_ops_int8.cpp
  # CPU NEON backend (modular)
  src/labneura/backends/cpu/neon/neon_backend.cpp
  src/labneura/backends/cpu/neon/neon_ops_fp32.cpp
  src/labneura/backends/cpu/neon/neon_ops_int8.cpp
  src/labneura/backends/cpu/neon/neon_ops_int16.cpp
  # CPU AVX2 backend (modular)
  src/labneura/backends/cpu/avx2/avx2_backend.cpp
  src/labneura/backends/cpu/avx2/avx2_ops_fp32.cpp
  src/labneura/backends/cpu/avx2/avx2_ops_int8.cpp
  src/labneura/backends/cpu/avx2/avx2_ops_fp16.cpp
  src/labneura/backends/cpu/avx2/avx2_ops_int16.cpp
)
set_target_properties(labneura PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(labneura PRIVATE slang::slang)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  # Check for AVX2 support
  check_cxx_compile_flag("-mavx2", LABNEURA_COMPILER_SUPPORTS_AVX2)
  if(LABNEURA_COMPILER_SUPPORTS_AVX2)
    # Apply AVX2 flags to all AVX2 backend files
    set_source_files_properties(
      src/labneura/backends/cpu/avx2/avx2_backend.cpp
      src/labneura/backends/cpu/avx2/avx2_ops_fp32.cpp
      src/labneura/backends/cpu/avx2/avx2_ops_int8.cpp
      src/labneura/backends/cpu/avx2/avx2_ops_fp16.cpp
      src/labneura/backends/cpu/avx2/avx2_ops_int16.cpp
      PROPERTIES COMPILE_FLAGS "-mavx2"
    )
    target_compile_definitions(labneura PRIVATE LABNEURA_ENABLE_AVX2=1)
    message(STATUS "LabNeura: AVX2 backend enabled (-mavx2)")
  else()
    message(STATUS "LabNeura: AVX2 backend disabled (compiler lacks -mavx2)")
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
    ${CMAKE_CURRENT_BINARY_DIR}/generated
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
if(NOT SKBUILD)
  add_executable(labneura_example examples/main.cpp)
  target_link_libraries(labneura_example PRIVATE labneura)
endif()

if(SKBUILD)
  set(Python_FIND_FRAMEWORK FIRST)
  find_package(Python COMPONENTS Interpreter REQUIRED)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c
      "import pathlib, sysconfig; candidates = [sysconfig.get_path('include'), sysconfig.get_config_var('INCLUDEPY') or '', str(pathlib.Path(sysconfig.get_path('stdlib')).parents[1] / 'include' / ('python' + sysconfig.get_config_var('VERSION')))]; print(next((path for path in candidates if path and (pathlib.Path(path) / 'Python.h').is_file()), ''))"
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

  add_library(labneura_python MODULE
    python/labneura_py.cpp
    python/eda_bindings.cpp
  )
  set_target_properties(labneura_python PROPERTIES
    PREFIX ""
    OUTPUT_NAME labneura
    SUFFIX "${CMAKE_SHARED_MODULE_SUFFIX}"
  )
  target_link_libraries(labneura_python PRIVATE pybind11::module labneura)
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