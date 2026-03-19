cmake_minimum_required(VERSION 3.18)
project(opal_two_phase CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --------------------------------------------------------------------------
# Find Python and pybind11
# --------------------------------------------------------------------------
set(OPAL_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
set(VENV      "${OPAL_ROOT}/external/venv")

set(Python3_EXECUTABLE "${VENV}/bin/python")
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

execute_process(
    COMMAND "${VENV}/bin/python" -c
            "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
find_package(pybind11 CONFIG REQUIRED)

# --------------------------------------------------------------------------
# opal_two_phase extension module
# --------------------------------------------------------------------------
pybind11_add_module(opal_two_phase
    solver.cpp
    hem_model.cpp
    five_eq_model.cpp
    bindings.cpp
)

target_include_directories(opal_two_phase PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

target_compile_options(opal_two_phase PRIVATE
    -O3
    -Wall
    -Wextra
    -Wpedantic
)

# Install the .so next to the source so tests can import it directly
install(TARGETS opal_two_phase
        LIBRARY DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}")

add_custom_target(install_inplace
    COMMAND ${CMAKE_COMMAND} --install . --prefix "${CMAKE_CURRENT_SOURCE_DIR}"
    DEPENDS opal_two_phase
    COMMENT "Installing opal_two_phase.so into solver/two_phase/"
)
