include_guard(GLOBAL)

function(_labneura_escape_liberty_chunk input_text output_variable)
  set(escaped "${input_text}")
  string(REPLACE "\\" "\\\\" escaped "${escaped}")
  string(REPLACE "\"" "\\\"" escaped "${escaped}")
  string(REPLACE "\t" "\\t" escaped "${escaped}")
  string(REPLACE "\r" "\\r" escaped "${escaped}")
  string(REPLACE "\n" "\\n" escaped "${escaped}")
  set(${output_variable} "${escaped}" PARENT_SCOPE)
endfunction()

function(labneura_generate_embedded_liberty input_path template_path output_path)
  if(NOT EXISTS "${input_path}")
    message(FATAL_ERROR "Liberty input does not exist: ${input_path}")
  endif()
  if(NOT EXISTS "${template_path}")
    message(FATAL_ERROR "Liberty header template does not exist: ${template_path}")
  endif()

  file(READ "${input_path}" liberty_source)
  string(LENGTH "${liberty_source}" liberty_length)
  set(chunk_length 4096)
  set(LABNEURA_EDA_LIBERTY_CHUNKS "")
  set(LABNEURA_EDA_LIBERTY_CHUNK_COUNT 0)

  set(offset 0)
  while(offset LESS liberty_length)
    math(EXPR remaining "${liberty_length} - ${offset}")
    if(remaining GREATER chunk_length)
      set(current_length ${chunk_length})
    else()
      set(current_length ${remaining})
    endif()
    string(SUBSTRING "${liberty_source}" ${offset} ${current_length} chunk)
    _labneura_escape_liberty_chunk("${chunk}" escaped_chunk)
    string(APPEND LABNEURA_EDA_LIBERTY_CHUNKS
      "    std::string_view(\"${escaped_chunk}\", ${current_length}),\n")
    math(EXPR LABNEURA_EDA_LIBERTY_CHUNK_COUNT
      "${LABNEURA_EDA_LIBERTY_CHUNK_COUNT} + 1")
    math(EXPR offset "${offset} + ${current_length}")
  endwhile()

  get_filename_component(output_directory "${output_path}" DIRECTORY)
  file(MAKE_DIRECTORY "${output_directory}")
  configure_file("${template_path}" "${output_path}" @ONLY)
endfunction()

if(DEFINED LABNEURA_EMBED_INPUT OR
   DEFINED LABNEURA_EMBED_TEMPLATE OR
   DEFINED LABNEURA_EMBED_OUTPUT)
  foreach(required_variable IN ITEMS
      LABNEURA_EMBED_INPUT LABNEURA_EMBED_TEMPLATE LABNEURA_EMBED_OUTPUT)
    if(NOT DEFINED ${required_variable} OR "${${required_variable}}" STREQUAL "")
      message(FATAL_ERROR "${required_variable} is required")
    endif()
  endforeach()
  labneura_generate_embedded_liberty(
    "${LABNEURA_EMBED_INPUT}"
    "${LABNEURA_EMBED_TEMPLATE}"
    "${LABNEURA_EMBED_OUTPUT}")
endif()
