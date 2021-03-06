
cmake_minimum_required (VERSION 3.3)
project(multigrid)

enable_testing()

include_directories(./include ./xtensor/include ./xtl/include)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

add_library(multigrid_poisson ./src/multigrid.cpp)
target_compile_options(multigrid_poisson PUBLIC -fPIC)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(multigrid ./src/bindpy.cpp)
target_link_libraries(multigrid PRIVATE multigrid_poisson)
set_target_properties(multigrid PROPERTIES
                      CXX_STANDARD 17
		      CXX_STANDARD_REQUIRED ON
		      CXX_EXTENSIONS OFF)

add_executable(multigrid_tester ./src/test.cpp)
target_link_libraries(multigrid_tester multigrid_poisson gtest pthread)
add_test(multigrid_test multigrid_tester)

configure_file(./scripts/functional.py functional.py COPYONLY)
configure_file(./scripts/part1.py part1.py COPYONLY)
configure_file(./scripts/part3.py part3.py COPYONLY)
configure_file(./scripts/part4.py part4.py COPYONLY)
