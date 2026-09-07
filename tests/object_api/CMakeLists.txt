cmake_minimum_required(VERSION 3.25)
project(anom_object_protocol_tests LANGUAGES C CXX)
include(CTest)

# Can be configured independently without OpenCV, FAISS, CUDA or ONNX Runtime.
get_filename_component(ANOM_OBJECT_SOURCE_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
add_library(anom_object_test_runtime SHARED
    "${ANOM_OBJECT_SOURCE_ROOT}/src/c_api/algorithm_objects.cpp"
    runtime_stubs.cpp)
target_compile_definitions(anom_object_test_runtime PRIVATE ANOM_ENGINE_BUILD=1)
target_include_directories(anom_object_test_runtime
    PUBLIC "${ANOM_OBJECT_SOURCE_ROOT}/include"
    PRIVATE "${ANOM_OBJECT_SOURCE_ROOT}/src")
target_compile_features(anom_object_test_runtime PRIVATE cxx_std_20)

add_executable(c_object_protocol_tests object_protocol_tests.c)
target_link_libraries(c_object_protocol_tests PRIVATE anom_object_test_runtime)
target_compile_features(c_object_protocol_tests PRIVATE c_std_11)

configure_file(object_protocol_tests.c "${CMAKE_CURRENT_BINARY_DIR}/object_protocol_tests.cpp" COPYONLY)
add_executable(cpp_object_protocol_tests "${CMAKE_CURRENT_BINARY_DIR}/object_protocol_tests.cpp")
target_link_libraries(cpp_object_protocol_tests PRIVATE anom_object_test_runtime)
target_compile_features(cpp_object_protocol_tests PRIVATE cxx_std_20)

# Each C header is compiled as the first include in its own translation unit.
add_library(c_object_headers OBJECT
    direct_header.c
    efficientad_header.c
    dfkde_header.c
    padim_header.c
    patchcore_header.c
    spade_header.c
    yolo_header.c)
target_include_directories(c_object_headers PRIVATE "${ANOM_OBJECT_SOURCE_ROOT}/include")
target_compile_features(c_object_headers PRIVATE c_std_11)
set_target_properties(c_object_protocol_tests c_object_headers PROPERTIES C_EXTENSIONS NO)
if(MSVC)
    foreach(target IN ITEMS anom_object_test_runtime c_object_protocol_tests cpp_object_protocol_tests c_object_headers)
        target_compile_options(${target} PRIVATE /W4 /WX /utf-8)
    endforeach()
    # The consumer deliberately uses another default convention.
    target_compile_options(c_object_protocol_tests PRIVATE /Gz)
    target_compile_options(cpp_object_protocol_tests PRIVATE /Gz)
endif()
add_test(NAME c_object_protocol_tests COMMAND c_object_protocol_tests)
add_test(NAME cpp_object_protocol_tests COMMAND cpp_object_protocol_tests)
