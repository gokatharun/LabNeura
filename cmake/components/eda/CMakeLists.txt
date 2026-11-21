# EDA component ownership. Included by the root build coordinator.

set(LABNEURA_EDA_FRONTEND_SOURCES
  src/labneura/eda/parser/verilog/syntax/native_verilog_parser.cpp
  src/labneura/eda/parser/verilog/lexer/verilog_lexer.cpp
  src/labneura/eda/parser/verilog/lexer/verilog_byte_classifier.cpp
  src/labneura/eda/parser/verilog/preprocessor/verilog_preprocessor.cpp
  src/labneura/eda/parser/verilog/syntax/verilog_syntax.cpp
  src/labneura/eda/parser/verilog/semantic/verilog_semantic.cpp
  src/labneura/eda/parser/verilog/semantic/verilog_callable_binding.cpp
  src/labneura/eda/parser/verilog/types/verilog_type.cpp
  src/labneura/eda/parser/verilog/evaluation/verilog_constant.cpp
  src/labneura/eda/parser/verilog/evaluation/verilog_function_evaluation.cpp
  src/labneura/eda/parser/verilog/evaluation/verilog_task_execution.cpp
  src/labneura/eda/parser/verilog/evaluation/verilog_loop_execution.cpp
  src/labneura/eda/parser/verilog/evaluation/verilog_branch_execution.cpp
  src/labneura/eda/parser/verilog/elaboration/verilog_elaboration.cpp
  src/labneura/eda/parser/verilog/elaboration/native_verilog_project_context.cpp
  src/labneura/eda/parser/verilog/elaboration/native_verilog_project_projection.cpp
  src/labneura/eda/parser/liberty/liberty_parser.cpp
)

set(LABNEURA_EDA_ANALYSIS_SOURCES
  src/labneura/eda/analysis/verilog_control_flow.cpp
  src/labneura/eda/analysis/verilog_process_analysis.cpp
  src/labneura/eda/analysis/verilog_process_dependencies.cpp
  src/labneura/eda/analysis/verilog_assignment_coverage.cpp
  src/labneura/eda/analysis/verilog_sequential_analysis.cpp
  src/labneura/eda/analysis/verilog_synthesis_eligibility.cpp
)

set(LABNEURA_EDA_LOWERING_SOURCES
  src/labneura/eda/lowering/verilog_lvalue.cpp
  src/labneura/eda/lowering/verilog_assignment.cpp
  src/labneura/eda/lowering/verilog_native_lowering.cpp
  src/labneura/eda/lowering/verilog_expression_lowering.cpp
  src/labneura/eda/lowering/verilog_combinational_lowering.cpp
  src/labneura/eda/lowering/verilog_sequential_lowering.cpp
  src/labneura/eda/lowering/verilog_module_lowering.cpp
)

set(LABNEURA_EDA_NETLIST_MAPPING_SOURCES
  src/labneura/eda/netlist/netlist_json.cpp
  src/labneura/eda/netlist/netlist_verilog.cpp
  src/labneura/eda/netlist/netlist_graph.cpp
  src/labneura/eda/netlist/netlist_blif.cpp
  src/labneura/eda/mapping/generic/generic_mapping.cpp
  src/labneura/eda/mapping/technology/technology_mapping.cpp
  src/labneura/eda/mapping/fpga/fpga_mapping.cpp
  src/labneura/eda/mapping/fpga/fpga_library.cpp
)

set(LABNEURA_EDA_SOURCES
  ${LABNEURA_EDA_FRONTEND_SOURCES}
  ${LABNEURA_EDA_ANALYSIS_SOURCES}
  ${LABNEURA_EDA_LOWERING_SOURCES}
  ${LABNEURA_EDA_NETLIST_MAPPING_SOURCES}
)

function(labneura_add_eda_component)
  set(default_liberty
    "${PROJECT_SOURCE_DIR}/assets/sky130_fd_sc_hd__tt_025C_1v80.lib")
  set(liberty_template
    "${PROJECT_SOURCE_DIR}/src/labneura/eda/parser/liberty/default_cell_library.h.in")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${default_liberty}"
    "${liberty_template}"
    "${PROJECT_SOURCE_DIR}/cmake/GenerateEmbeddedLiberty.cmake")
  include("${PROJECT_SOURCE_DIR}/cmake/GenerateEmbeddedLiberty.cmake")
  labneura_generate_embedded_liberty(
    "${default_liberty}"
    "${liberty_template}"
    "${CMAKE_CURRENT_BINARY_DIR}/generated/labneura/eda/default_cell_library.h")

  add_library(labneura_eda_objects OBJECT ${LABNEURA_EDA_SOURCES})
  if(LABNEURA_VERILOG_LEXER_HAS_AVX2)
    set(avx2_source
      src/labneura/eda/parser/verilog/lexer/verilog_byte_classifier_avx2.cpp)
    target_sources(labneura_eda_objects PRIVATE "${avx2_source}")
    set_source_files_properties("${avx2_source}"
      PROPERTIES COMPILE_OPTIONS "${LABNEURA_VERILOG_LEXER_AVX2_FLAG}")
  endif()
  set_target_properties(labneura_eda_objects PROPERTIES
    POSITION_INDEPENDENT_CODE ON)
  target_include_directories(labneura_eda_objects
    PUBLIC
      $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>
    PRIVATE
      ${PROJECT_SOURCE_DIR}/src
      ${CMAKE_CURRENT_BINARY_DIR}/generated)
  if(LABNEURA_VERILOG_LEXER_HAS_NEON)
    target_compile_definitions(labneura_eda_objects
      PRIVATE LABNEURA_VERILOG_LEXER_HAS_NEON=1)
    message(STATUS "Verilog lexer NEON byte classification enabled")
  else()
    message(STATUS "Verilog lexer NEON byte classification unavailable; scalar enabled")
  endif()
  if(LABNEURA_VERILOG_LEXER_HAS_AVX2)
    target_compile_definitions(labneura_eda_objects
      PRIVATE LABNEURA_VERILOG_LEXER_HAS_AVX2=1)
    message(STATUS "Verilog lexer AVX2 byte classification compiled; runtime CPU check enabled")
  else()
    message(STATUS "Verilog lexer AVX2 byte classification unavailable; scalar fallback enabled")
  endif()

  add_library(labneura_eda STATIC $<TARGET_OBJECTS:labneura_eda_objects>)
  set_target_properties(labneura_eda PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    EXPORT_NAME EDA)
  target_include_directories(labneura_eda
    PUBLIC
      $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>)
  target_compile_features(labneura_eda PUBLIC cxx_std_20)
  set(LABNEURA_OBJECT_TARGETS
    ${LABNEURA_OBJECT_TARGETS} labneura_eda_objects PARENT_SCOPE)
endfunction()
