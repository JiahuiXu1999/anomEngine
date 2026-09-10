cmake_minimum_required(VERSION 3.25)
project(anom_backend_memory_tests LANGUAGES CXX)
include(CTest)
find_package(Threads REQUIRED)
get_filename_component(ANOM_MEMORY_SOURCE_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
# Exercise the production loader and Tensor implementation without GPU/ORT/OpenCV.
add_library(memory_backend_plugin MODULE memory_backend_plugin.cpp)
set_target_properties(memory_backend_plugin PROPERTIES
    PREFIX "" OUTPUT_NAME "anom_backend_onnxruntime"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/plugins"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/plugins"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/plugins")
add_executable(backend_memory_tests backend_memory_tests.cpp
    "${ANOM_MEMORY_SOURCE_ROOT}/src/backends/backend_factory.cpp"
    "${ANOM_MEMORY_SOURCE_ROOT}/src/plugins/dynamic_library.cpp"
    "${ANOM_MEMORY_SOURCE_ROOT}/src/model/types.cpp")
target_link_libraries(backend_memory_tests PRIVATE Threads::Threads ${CMAKE_DL_LIBS})
set_target_properties(backend_memory_tests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin")
target_compile_definitions(backend_memory_tests PRIVATE
    ANOM_MEMORY_PLUGIN_DIR="$<TARGET_FILE_DIR:memory_backend_plugin>")
add_dependencies(backend_memory_tests memory_backend_plugin)
foreach(target IN ITEMS memory_backend_plugin backend_memory_tests)
    target_include_directories(${target} PRIVATE "${ANOM_MEMORY_SOURCE_ROOT}/src")
    target_compile_features(${target} PRIVATE cxx_std_20)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /WX /utf-8 /permissive-)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif()
endforeach()
add_test(NAME backend_memory_tests COMMAND backend_memory_tests)
