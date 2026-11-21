# Tensor public implementation and internal Tensor Core source ownership.
set(LABNEURA_TENSOR_SOURCES
  ${PROJECT_SOURCE_DIR}/src/labneura/tensor/tensor.cpp
  ${PROJECT_SOURCE_DIR}/src/labneura/tensor/core/runtime/runtime.cpp
  ${PROJECT_SOURCE_DIR}/src/labneura/tensor/core/execution/parallel.cpp
  ${PROJECT_SOURCE_DIR}/src/labneura/tensor/core/storage/tensor_storage.cpp
  ${PROJECT_SOURCE_DIR}/src/labneura/tensor/core/dispatch/backend_factory.cpp
  ${PROJECT_SOURCE_DIR}/src/labneura/tensor/core/kernels/cpu/common/base.cpp
  ${PROJECT_SOURCE_DIR}/src/labneura/tensor/core/kernels/cpu/generic/generic_backend.cpp
  ${PROJECT_SOURCE_DIR}/src/labneura/tensor/core/kernels/cpu/generic/generic_ops_fp32.cpp
  ${PROJECT_SOURCE_DIR}/src/labneura/tensor/core/kernels/cpu/generic/generic_ops_compare.cpp
  ${PROJECT_SOURCE_DIR}/src/labneura/tensor/core/kernels/cpu/generic/generic_ops_logical.cpp
  ${PROJECT_SOURCE_DIR}/src/labneura/tensor/core/kernels/cpu/generic/generic_ops_int8.cpp
  ${PROJECT_SOURCE_DIR}/src/labneura/tensor/core/kernels/cpu/neon/neon_backend.cpp
  ${PROJECT_SOURCE_DIR}/src/labneura/tensor/core/kernels/cpu/neon/neon_ops_compare.cpp
  ${PROJECT_SOURCE_DIR}/src/labneura/tensor/core/kernels/cpu/neon/neon_ops_logical.cpp
  ${PROJECT_SOURCE_DIR}/src/labneura/tensor/core/kernels/cpu/neon/neon_ops_fp32.cpp
  ${PROJECT_SOURCE_DIR}/src/labneura/tensor/core/kernels/cpu/neon/neon_ops_int8.cpp
  ${PROJECT_SOURCE_DIR}/src/labneura/tensor/core/kernels/cpu/neon/neon_ops_int16.cpp
  ${PROJECT_SOURCE_DIR}/src/labneura/tensor/core/kernels/cpu/avx2/avx2_backend.cpp
  ${PROJECT_SOURCE_DIR}/src/labneura/tensor/core/kernels/cpu/avx2/avx2_ops_compare.cpp
  ${PROJECT_SOURCE_DIR}/src/labneura/tensor/core/kernels/cpu/avx2/avx2_ops_logical.cpp
  ${PROJECT_SOURCE_DIR}/src/labneura/tensor/core/kernels/cpu/avx2/avx2_ops_fp32.cpp
  ${PROJECT_SOURCE_DIR}/src/labneura/tensor/core/kernels/cpu/avx2/avx2_ops_int8.cpp
  ${PROJECT_SOURCE_DIR}/src/labneura/tensor/core/kernels/cpu/avx2/avx2_ops_fp16.cpp
  ${PROJECT_SOURCE_DIR}/src/labneura/tensor/core/kernels/cpu/avx2/avx2_ops_int16.cpp
)

set(LABNEURA_TENSOR_AVX2_CONSTRUCTOR_SOURCE
  ${PROJECT_SOURCE_DIR}/src/labneura/tensor/core/kernels/cpu/avx2/avx2_backend_constructor.cpp
)
set(LABNEURA_TENSOR_F16C_SOURCE
  ${PROJECT_SOURCE_DIR}/src/labneura/tensor/core/kernels/cpu/avx2/avx2_ops_fp16_f16c.cpp
)

set(LABNEURA_TENSOR_GENERIC_IEEE_SOURCES
  ${PROJECT_SOURCE_DIR}/src/labneura/tensor/core/kernels/cpu/generic/generic_ops_fp32.cpp
  ${PROJECT_SOURCE_DIR}/src/labneura/tensor/core/kernels/cpu/generic/generic_ops_compare.cpp
)
set(LABNEURA_TENSOR_NEON_IEEE_SOURCES
  ${PROJECT_SOURCE_DIR}/src/labneura/tensor/core/kernels/cpu/neon/neon_ops_compare.cpp
  ${PROJECT_SOURCE_DIR}/src/labneura/tensor/core/kernels/cpu/neon/neon_ops_fp32.cpp
)

set(LABNEURA_TENSOR_PUBLIC_HEADERS
  ${PROJECT_SOURCE_DIR}/include/labneura/tensor/tensor.h
  ${PROJECT_SOURCE_DIR}/include/labneura/tensor/runtime.h
  ${PROJECT_SOURCE_DIR}/include/labneura/tensor/types.hpp
)

function(labneura_add_tensor_component)
  set(labneura_tensor_target_x86 FALSE)
  set(labneura_tensor_target_arm FALSE)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64" OR
     CMAKE_OSX_ARCHITECTURES MATCHES "(^|;)x86_64($|;)")
    set(labneura_tensor_target_x86 TRUE)
  endif()
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64" OR
     CMAKE_OSX_ARCHITECTURES MATCHES "(^|;)arm64($|;)")
    set(labneura_tensor_target_arm TRUE)
  endif()
  if(CMAKE_OSX_ARCHITECTURES MATCHES "(^|;)x86_64($|;)")
    set(labneura_tensor_target_arm FALSE)
  elseif(CMAKE_OSX_ARCHITECTURES MATCHES "(^|;)arm64($|;)")
    set(labneura_tensor_target_x86 FALSE)
  endif()

  add_library(labneura_tensor_objects OBJECT ${LABNEURA_TENSOR_SOURCES})
  add_library(labneura_tensor_core ALIAS labneura_tensor_objects)
  set_target_properties(labneura_tensor_objects PROPERTIES
    POSITION_INDEPENDENT_CODE ON)
  target_include_directories(labneura_tensor_objects
    PUBLIC
      $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>
    PRIVATE ${PROJECT_SOURCE_DIR}/src)

  add_library(labneura_tensor STATIC $<TARGET_OBJECTS:labneura_tensor_objects>)
  set_target_properties(labneura_tensor PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    EXPORT_NAME Tensor)
  target_include_directories(labneura_tensor
    PUBLIC
      $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>)
  target_compile_features(labneura_tensor PUBLIC cxx_std_20)
  if(TARGET labneura_core)
    target_link_libraries(labneura_tensor PUBLIC labneura_core)
  endif()

  if(labneura_tensor_target_x86)
    if(MSVC)
      check_cxx_compiler_flag("/arch:AVX2" LABNEURA_COMPILER_SUPPORTS_AVX2)
      set(LABNEURA_TENSOR_AVX2_FLAGS "/arch:AVX2")
      set(LABNEURA_TENSOR_AVX2_IEEE_FLAGS "/arch:AVX2 /fp:precise")
      set(LABNEURA_TENSOR_F16C_FLAGS "/arch:AVX2 /fp:precise")
    else()
      check_cxx_compiler_flag("-mavx2" LABNEURA_COMPILER_SUPPORTS_AVX2)
      set(LABNEURA_TENSOR_AVX2_FLAGS "-mavx2")
      set(LABNEURA_TENSOR_AVX2_IEEE_FLAGS "-mavx2 -fno-fast-math")
      set(LABNEURA_TENSOR_F16C_FLAGS "-mavx2 -mf16c -fno-fast-math")
    endif()
    if(LABNEURA_COMPILER_SUPPORTS_AVX2)
      target_sources(labneura_tensor_objects PRIVATE
        ${LABNEURA_TENSOR_AVX2_CONSTRUCTOR_SOURCE})
      set_source_files_properties(
        ${PROJECT_SOURCE_DIR}/src/labneura/tensor/core/kernels/cpu/avx2/avx2_backend.cpp
        ${PROJECT_SOURCE_DIR}/src/labneura/tensor/core/kernels/cpu/avx2/avx2_ops_compare.cpp
        ${PROJECT_SOURCE_DIR}/src/labneura/tensor/core/kernels/cpu/avx2/avx2_ops_logical.cpp
        ${PROJECT_SOURCE_DIR}/src/labneura/tensor/core/kernels/cpu/avx2/avx2_ops_int8.cpp
        ${PROJECT_SOURCE_DIR}/src/labneura/tensor/core/kernels/cpu/avx2/avx2_ops_int16.cpp
        PROPERTIES COMPILE_FLAGS "${LABNEURA_TENSOR_AVX2_FLAGS}")
      set_source_files_properties(
        ${PROJECT_SOURCE_DIR}/src/labneura/tensor/core/kernels/cpu/avx2/avx2_ops_fp32.cpp
        ${PROJECT_SOURCE_DIR}/src/labneura/tensor/core/kernels/cpu/avx2/avx2_ops_fp16.cpp
        ${PROJECT_SOURCE_DIR}/src/labneura/tensor/core/kernels/cpu/avx2/avx2_ops_compare.cpp
        PROPERTIES COMPILE_FLAGS "${LABNEURA_TENSOR_AVX2_IEEE_FLAGS}")
      target_compile_definitions(labneura_tensor_objects
        PRIVATE LABNEURA_ENABLE_AVX2=1)
      message(STATUS "LabNeura: AVX2 backend enabled (${LABNEURA_TENSOR_AVX2_FLAGS})")

      if(MSVC)
        set(LABNEURA_TENSOR_REQUIRED_FLAGS_SAVED "${CMAKE_REQUIRED_FLAGS}")
        set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} /arch:AVX2")
        check_cxx_source_compiles(
          "#include <immintrin.h>
           int main() {
             const __m128i half = _mm_setzero_si128();
             const __m256 value = _mm256_cvtph_ps(half);
             return _mm256_movemask_ps(value);
           }"
          LABNEURA_COMPILER_SUPPORTS_F16C)
        set(CMAKE_REQUIRED_FLAGS "${LABNEURA_TENSOR_REQUIRED_FLAGS_SAVED}")
      else()
        check_cxx_compiler_flag("-mf16c" LABNEURA_COMPILER_SUPPORTS_F16C)
      endif()
      if(LABNEURA_COMPILER_SUPPORTS_F16C)
        target_sources(labneura_tensor_objects PRIVATE ${LABNEURA_TENSOR_F16C_SOURCE})
        set_source_files_properties(${LABNEURA_TENSOR_F16C_SOURCE}
          PROPERTIES COMPILE_FLAGS "${LABNEURA_TENSOR_F16C_FLAGS}")
        target_compile_definitions(labneura_tensor_objects
          PRIVATE LABNEURA_ENABLE_AVX2_F16C=1)
        message(STATUS "LabNeura: AVX2 FP16 F16C kernel enabled (isolated)")
      endif()
    else()
      message(STATUS "LabNeura: AVX2 backend disabled (compiler lacks the required ISA flag)")
    endif()
  endif()

  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    set_source_files_properties(${LABNEURA_TENSOR_GENERIC_IEEE_SOURCES}
      PROPERTIES COMPILE_FLAGS "-fno-fast-math")
  endif()
  if(labneura_tensor_target_arm)
    target_compile_definitions(labneura_tensor_objects PRIVATE LABNEURA_ENABLE_NEON=1)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
      set_source_files_properties(${LABNEURA_TENSOR_NEON_IEEE_SOURCES}
        PROPERTIES COMPILE_FLAGS "-fno-fast-math")
    endif()
  endif()

  if(BUILD_TESTING AND NOT SKBUILD)
    add_executable(labneura_tensor_component_smoke_test
      ${PROJECT_SOURCE_DIR}/tests/tensor/component_smoke_test.cpp)
    target_link_libraries(labneura_tensor_component_smoke_test PRIVATE labneura_tensor)
    add_test(NAME tensor_component_smoke COMMAND labneura_tensor_component_smoke_test)
  endif()

  set(LABNEURA_OBJECT_TARGETS
    ${LABNEURA_OBJECT_TARGETS} labneura_tensor_objects PARENT_SCOPE)
endfunction()
