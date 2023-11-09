project(qd_benchmark) # Replace 'my_project' with the name of your project
cmake_minimum_required(VERSION 3.15...3.27)
find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(TBB REQUIRED)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Detect the installed nanobind package and import it into CMake
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ext/nanobind)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

# Build the core parts of nanobind once
nanobind_build_library(nanobind SHARED)

# Compile an extension library
add_library(pf_mapelites MODULE pf_mapelites.cpp)

# .. and link it against the nanobind parts
target_link_libraries(pf_mapelites PRIVATE nanobind Eigen3::Eigen TBB::tbb)

# .. enable size optimizations
#nanobind_opt_size(my_ext) # no size optimization, we want -O3

# .. enable link time optimization
nanobind_lto(pf_mapelites)

# .. set the default symbol visibility to 'hidden'
nanobind_set_visibility(pf_mapelites)

# .. strip unneeded symbols and debug info from the binary (only active in release builds)
nanobind_strip(pf_mapelites)

# .. disable the stack protector
nanobind_disable_stack_protector(pf_mapelites)

# .. set the Python extension suffix
nanobind_extension(pf_mapelites)

# .. set important compilation flags
nanobind_compile_options(pf_mapelites)

# .. set important linker flags
nanobind_link_options(pf_mapelites)

# Statically link against libstdc++/libgcc when targeting musllinux
nanobind_musl_static_libcpp(pf_mapelites)

# nanobind_add_module(pf_mapelites pf_mapelites.cpp)

# add_library(fit_gp STATIC fit_gp.cpp)
# target_link_libraries (fit_gp Eigen3::Eigen)
# set_target_properties(fit_gp PROPERTIES POSITION_INDEPENDENT_CODE ON)
# set_property(TARGET fit_gp PROPERTY CXX_STANDARD 17)



# target_link_libraries(pf_mapelites PRIVATE fit_gp)