# Core owns domain-independent hardware and platform infrastructure.
set(LABNEURA_CORE_SOURCES
  ${PROJECT_SOURCE_DIR}/src/labneura/core/cpu_features.cpp
)

set(LABNEURA_CORE_PUBLIC_HEADERS
  ${PROJECT_SOURCE_DIR}/include/labneura/core/alignment.hpp
  ${PROJECT_SOURCE_DIR}/include/labneura/core/cpu_features.h
)

function(labneura_add_core_component)
  add_library(labneura_core_objects OBJECT ${LABNEURA_CORE_SOURCES})
  add_library(labneura_accel_core_objects ALIAS labneura_core_objects)
  set_target_properties(labneura_core_objects PROPERTIES
    POSITION_INDEPENDENT_CODE ON)
  target_include_directories(labneura_core_objects
    PUBLIC
      $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>
    PRIVATE ${PROJECT_SOURCE_DIR}/src)

  add_library(labneura_core STATIC $<TARGET_OBJECTS:labneura_core_objects>)
  add_library(labneura_accel_core ALIAS labneura_core)
  set_target_properties(labneura_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    EXPORT_NAME AccelCore)
  target_include_directories(labneura_core
    PUBLIC
      $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>)
  target_compile_features(labneura_core PUBLIC cxx_std_20)

  if(BUILD_TESTING AND NOT SKBUILD)
    add_executable(labneura_core_cpu_features_test
      ${PROJECT_SOURCE_DIR}/tests/core/cpu_features_test.cpp)
    target_link_libraries(labneura_core_cpu_features_test PRIVATE labneura_core)
    add_test(NAME core_cpu_features COMMAND labneura_core_cpu_features_test)
  endif()

  set(LABNEURA_OBJECT_TARGETS
    ${LABNEURA_OBJECT_TARGETS} labneura_core_objects PARENT_SCOPE)
endfunction()
