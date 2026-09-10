cmake_minimum_required(VERSION 3.25)
project(anom_faiss_routing_tests LANGUAGES CXX)
include(CTest)
find_package(Threads REQUIRED)
get_filename_component(ANOM_FAISS_TEST_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
set(ANOM_FAISS_TEST_SOURCES
    "${ANOM_FAISS_TEST_ROOT}/src/retrieval/faiss_index.cpp"
    "${ANOM_FAISS_TEST_ROOT}/src/retrieval/gpu_faiss_client.cpp"
    "${ANOM_FAISS_TEST_ROOT}/src/plugins/dynamic_library.cpp")
add_library(fake_faiss_cuda_plugin MODULE "${ANOM_FAISS_TEST_ROOT}/src/plugins/faiss_cuda_plugin.cpp")
set_target_properties(fake_faiss_cuda_plugin PROPERTIES PREFIX "" OUTPUT_NAME "anom_search_faiss_cuda"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/fake_plugins"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/fake_plugins"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/fake_plugins")
add_executable(faiss_routing_tests faiss_search_tests.cpp ${ANOM_FAISS_TEST_SOURCES})
target_compile_definitions(faiss_routing_tests PRIVATE ANOM_FAISS_TEST_FAKE=1
    ANOM_FAISS_TEST_PLUGIN_DIR="$<TARGET_FILE_DIR:fake_faiss_cuda_plugin>")
add_dependencies(faiss_routing_tests fake_faiss_cuda_plugin)
target_link_libraries(faiss_routing_tests PRIVATE Threads::Threads ${CMAKE_DL_LIBS})
foreach(target IN ITEMS fake_faiss_cuda_plugin faiss_routing_tests)
    target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_LIST_DIR}/fakes" "${ANOM_FAISS_TEST_ROOT}/src")
    target_compile_features(${target} PRIVATE cxx_std_20)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /WX /utf-8 /permissive- /UNDEBUG)
        target_compile_definitions(${target} PRIVATE _CRT_SECURE_NO_WARNINGS)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Werror -UNDEBUG)
    endif()
endforeach()
add_test(NAME faiss_routing_tests COMMAND faiss_routing_tests)

# Compile the host and both real algorithm adapters against declaration-only
# image headers. This catches integration errors without claiming image QA.
add_library(faiss_adapter_compile OBJECT
    "${ANOM_FAISS_TEST_ROOT}/src/adapters/adapter.cpp"
    "${ANOM_FAISS_TEST_ROOT}/src/model/inference_session.cpp"
    "${ANOM_FAISS_TEST_ROOT}/src/adapters/patchcore_adapter.cpp"
    "${ANOM_FAISS_TEST_ROOT}/src/adapters/spade_adapter.cpp")
add_library(faiss_algorithm_compile OBJECT "${ANOM_FAISS_TEST_ROOT}/src/plugins/algorithm_plugin.cpp")
target_compile_definitions(faiss_algorithm_compile PRIVATE ANOM_ALGORITHM_PATCHCORE=1 ANOM_ALGORITHM_NAME="patchcore")
foreach(target IN ITEMS faiss_adapter_compile faiss_algorithm_compile)
    target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_LIST_DIR}/fakes" "${ANOM_FAISS_TEST_ROOT}/src")
    target_compile_features(${target} PRIVATE cxx_std_20)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /WX /utf-8 /permissive-)
        target_compile_definitions(${target} PRIVATE _CRT_SECURE_NO_WARNINGS)
    endif()
endforeach()

# Same test body with genuine Faiss/CUDA when configured by the full project.
if(TARGET anom_search_faiss_cuda)
    add_executable(faiss_gpu_parity_tests faiss_search_tests.cpp ${ANOM_FAISS_TEST_SOURCES})
    target_include_directories(faiss_gpu_parity_tests PRIVATE "${ANOM_FAISS_TEST_ROOT}/src")
    target_compile_features(faiss_gpu_parity_tests PRIVATE cxx_std_20)
    target_compile_definitions(faiss_gpu_parity_tests PRIVATE ANOM_FAISS_TEST_FAKE=0
        ANOM_FAISS_TEST_PLUGIN_DIR="$<TARGET_FILE_DIR:anom_search_faiss_cuda>")
    target_link_libraries(faiss_gpu_parity_tests PRIVATE anom::faiss Threads::Threads ${CMAKE_DL_LIBS})
    add_dependencies(faiss_gpu_parity_tests anom_search_faiss_cuda)
    add_test(NAME faiss_gpu_parity_tests COMMAND faiss_gpu_parity_tests)
    set_tests_properties(faiss_gpu_parity_tests PROPERTIES SKIP_RETURN_CODE 77)
endif()
