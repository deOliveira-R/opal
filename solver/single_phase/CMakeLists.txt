cmake_minimum_required(VERSION 3.18)
project(opal_single_phase CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --------------------------------------------------------------------------
# Find Python and pybind11
# --------------------------------------------------------------------------
# Point at the OPAL project venv so pybind11 is found automatically.
set(OPAL_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
set(VENV      "${OPAL_ROOT}/external/venv")

# Tell CMake which Python to use (the venv interpreter)
set(Python3_EXECUTABLE "${VENV}/bin/python")
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Find pybind11 via the venv installation
execute_process(
    COMMAND "${VENV}/bin/python" -c
            "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
find_package(pybind11 CONFIG REQUIRED)

# --------------------------------------------------------------------------
# opal_single_phase extension module
# --------------------------------------------------------------------------
pybind11_add_module(opal_single_phase
    solver.cpp
    bindings.cpp
)

target_include_directories(opal_single_phase PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

target_compile_options(opal_single_phase PRIVATE
    -O3
    -Wall
    -Wextra
    -Wpedantic
)

# Install the .so next to the source so the driver can import it directly
install(TARGETS opal_single_phase
        LIBRARY DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}")

# --------------------------------------------------------------------------
# Convenience target: build and place .so in the source tree
# --------------------------------------------------------------------------
add_custom_target(install_inplace
    COMMAND ${CMAKE_COMMAND} --install . --prefix "${CMAKE_CURRENT_SOURCE_DIR}"
    DEPENDS opal_single_phase
    COMMENT "Installing opal_single_phase.so into solver/single_phase/"
)
