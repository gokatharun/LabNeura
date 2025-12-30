cmake_minimum_required(VERSION 3.14)
project(LabNeura LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set up libomp from Homebrew (for Apple Silicon M1)
if(APPLE)
  # Check for libomp installed via Homebrew
  if(EXISTS "/opt/homebrew/opt/libomp")
    set(LIBOMP_PATH "/opt/homebrew/opt/libomp")
    message(STATUS "Found libomp at: ${LIBOMP_PATH}")
  elseif(EXISTS "/usr/local/opt/libomp")
    set(LIBOMP_PATH "/usr/local/opt/libomp")
    message(STATUS "Found libomp at: ${LIBOMP_PATH}")
  else()
    message(FATAL_ERROR "libomp not found! Install with: brew install libomp")
  endif()
  
  # Configure include and link paths
  include_directories("${LIBOMP_PATH}/include")
  link_directories("${LIBOMP_PATH}/lib")
  
  # Set compiler and linker flags
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xclang -fopenmp")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lomp")
endif()

# Detect Apple Silicon M1
if(APPLE)
  execute_process(COMMAND uname -m OUTPUT_VARIABLE ARCH OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(ARCH MATCHES "arm64|aarch64")
    add_compile_definitions(LABNEURA_HAVE_M1=1)
    message(STATUS "Apple Silicon M1 detected, enabling NEON optimizations")
  endif()
endif()

# Core library with tensor implementation
add_library(labneura 
  src/labneura/tensor.cpp
)

target_include_directories(labneura
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Example executable
add_executable(labneura_example examples/main.cpp)
target_link_libraries(labneura_example PRIVATE labneura)

install(TARGETS labneura EXPORT LabNeuraTargets)
install(DIRECTORY include/ DESTINATION include)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/LabNeuraConfigVersion.cmake"
  VERSION 0.1
  COMPATIBILITY AnyNewerVersion
)
