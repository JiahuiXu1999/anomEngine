cmake_minimum_required(VERSION 3.25)

project(
    anomEngine
    VERSION 1.0.0
    DESCRIPTION "Cross-platform anomaly-detection inference engine"
    LANGUAGES C CXX)

include(CTest)
include(GNUInstallDirs)

# Project policy ----------------------------------------------------------------

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)

if(POLICY CMP0091)
    cmake_policy(SET CMP0091 NEW)
endif()
if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY
        "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL"
        CACHE STRING "MSVC runtime library" FORCE)
endif()

option(ANOM_ENABLE_TENSORRT "Build the TensorRT backend plugin" ON)
option(ANOM_ENABLE_ONNXRUNTIME "Build the ONNX Runtime backend plugin" ON)
option(ANOM_ENABLE_FAISS_GPU "Build the optional GPU Faiss retrieval plugin" OFF)
option(ANOM_BUILD_ALGO_DIRECT "Build the direct-prediction algorithm plugin" ON)
option(ANOM_BUILD_ALGO_EFFICIENTAD "Build the EfficientAD algorithm plugin" ON)
option(ANOM_BUILD_ALGO_DFKDE "Build the DFKDE algorithm plugin" ON)
option(ANOM_BUILD_ALGO_PADIM "Build the PaDiM algorithm plugin" ON)
option(ANOM_BUILD_ALGO_PATCHCORE "Build the PatchCore algorithm plugin" ON)
option(ANOM_BUILD_ALGO_SPADE "Build the SPADE algorithm plugin" ON)
option(ANOM_BUILD_ALGO_YOLO "Build the YOLO algorithm plugin" ON)
option(ANOM_ENABLE_STRICT_WARNINGS "Enable the project's strict warning set" ON)

set(ANOM_SOURCE_ROOT "${PROJECT_SOURCE_DIR}")
set(ANOM_PUBLIC_INCLUDE_ROOT "${ANOM_SOURCE_ROOT}/include")
set(ANOM_PRIVATE_SOURCE_ROOT "${ANOM_SOURCE_ROOT}/src")
set(ANOM_TEST_ROOT "${ANOM_SOURCE_ROOT}/tests")
set(ANOM_BUNDLED_DEPENDENCY_ROOT "${ANOM_SOURCE_ROOT}/libs")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")

if(UNIX AND NOT APPLE)
    set(ANOM_ORIGIN_RPATH "$ORIGIN")
else()
    set(ANOM_ORIGIN_RPATH "")
endif()

function(anom_assert_absolute_path variable_name)
    if(DEFINED ${variable_name} AND NOT "${${variable_name}}" STREQUAL "" AND
       NOT IS_ABSOLUTE "${${variable_name}}")
        message(FATAL_ERROR "${variable_name} must be an absolute path: '${${variable_name}}'")
    endif()
endfunction()

function(anom_apply_project_defaults target_name)
    target_compile_features(${target_name} PRIVATE cxx_std_20)
    set_target_properties(${target_name} PROPERTIES
        CXX_EXTENSIONS NO
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN YES)
    if(ANOM_ENABLE_STRICT_WARNINGS)
        if(MSVC)
            target_compile_options(${target_name} PRIVATE
                /W4 /utf-8 /permissive- /Zc:__cplusplus)
        elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(${target_name} PRIVATE
                -Wall -Wextra -Wpedantic)
        endif()
    elseif(MSVC)
        target_compile_options(${target_name} PRIVATE /utf-8)
    endif()
endfunction()

# Third-party dependencies ------------------------------------------------------

if(WIN32)
    set(_anom_default_opencv_root "${ANOM_BUNDLED_DEPENDENCY_ROOT}/opencv/build")
    set(_anom_default_faiss_root "${ANOM_BUNDLED_DEPENDENCY_ROOT}/faiss")
    set(_anom_default_onnxruntime_root "${ANOM_BUNDLED_DEPENDENCY_ROOT}/onnxruntime")
    set(_anom_default_tensorrt_root "${ANOM_BUNDLED_DEPENDENCY_ROOT}/TensorRT-10.16.1.11")
else()
    set(_anom_default_opencv_root "")
    set(_anom_default_faiss_root "")
    set(_anom_default_onnxruntime_root "")
    set(_anom_default_tensorrt_root "")
endif()

set(ANOM_OPENCV_ROOT "${_anom_default_opencv_root}" CACHE PATH
    "Absolute directory containing OpenCVConfig.cmake")
set(ANOM_FAISS_ROOT "${_anom_default_faiss_root}" CACHE PATH
    "Absolute root of the FAISS C/C++ distribution")
set(ANOM_ONNXRUNTIME_ROOT "${_anom_default_onnxruntime_root}" CACHE PATH
    "Absolute root of the ONNX Runtime C/C++ distribution")
set(ANOM_TENSORRT_ROOT "${_anom_default_tensorrt_root}" CACHE PATH
    "Absolute root of the TensorRT C++ distribution")

foreach(_anom_path_variable IN ITEMS
        ANOM_OPENCV_ROOT ANOM_FAISS_ROOT ANOM_ONNXRUNTIME_ROOT ANOM_TENSORRT_ROOT)
    anom_assert_absolute_path(${_anom_path_variable})
endforeach()

if(ANOM_OPENCV_ROOT)
    find_package(OpenCV CONFIG REQUIRED COMPONENTS core imgproc
        PATHS "${ANOM_OPENCV_ROOT}" NO_DEFAULT_PATH)
else()
    find_package(OpenCV CONFIG REQUIRED COMPONENTS core imgproc)
endif()

add_library(anom_opencv INTERFACE)
add_library(anom::opencv ALIAS anom_opencv)
if(TARGET opencv_core AND TARGET opencv_imgproc)
    target_link_libraries(anom_opencv INTERFACE opencv_core opencv_imgproc)
else()
    target_include_directories(anom_opencv SYSTEM INTERFACE "${OpenCV_INCLUDE_DIRS}")
    target_link_libraries(anom_opencv INTERFACE ${OpenCV_LIBS})
endif()

set(ANOM_NEEDS_FAISS OFF)
if(ANOM_BUILD_ALGO_PATCHCORE OR ANOM_BUILD_ALGO_SPADE OR BUILD_TESTING OR ANOM_ENABLE_FAISS_GPU)
    set(ANOM_NEEDS_FAISS ON)
endif()

if(ANOM_NEEDS_FAISS)
    set(_anom_faiss_include_hints "")
    set(_anom_faiss_library_hints "")
    if(ANOM_FAISS_ROOT)
        list(APPEND _anom_faiss_include_hints
            "${ANOM_FAISS_ROOT}/include" "${ANOM_FAISS_ROOT}")
        list(APPEND _anom_faiss_library_hints
            "${ANOM_FAISS_ROOT}/lib" "${ANOM_FAISS_ROOT}/lib64")
    endif()
    find_path(ANOM_FAISS_INCLUDE_DIR NAMES faiss/Index.h
        HINTS ${_anom_faiss_include_hints})
    find_library(ANOM_FAISS_LIBRARY NAMES faiss
        HINTS ${_anom_faiss_library_hints})
    find_library(ANOM_FAISS_C_LIBRARY NAMES faiss_c
        HINTS ${_anom_faiss_library_hints})
    if(NOT ANOM_FAISS_INCLUDE_DIR OR NOT ANOM_FAISS_LIBRARY OR NOT ANOM_FAISS_C_LIBRARY)
        message(FATAL_ERROR
            "FAISS was requested but was not found. Set ANOM_FAISS_ROOT to an absolute SDK path.")
    endif()
    add_library(anom_faiss UNKNOWN IMPORTED GLOBAL)
    add_library(anom::faiss ALIAS anom_faiss)
    set_target_properties(anom_faiss PROPERTIES
        IMPORTED_LOCATION "${ANOM_FAISS_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${ANOM_FAISS_INCLUDE_DIR}")
    add_library(anom_faiss_c UNKNOWN IMPORTED GLOBAL)
    add_library(anom::faiss_c ALIAS anom_faiss_c)
    set_target_properties(anom_faiss_c PROPERTIES
        IMPORTED_LOCATION "${ANOM_FAISS_C_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${ANOM_FAISS_INCLUDE_DIR}")
endif()

if(ANOM_ENABLE_TENSORRT)
    set(_anom_tensorrt_include_hints "")
    set(_anom_tensorrt_library_hints "")
    if(ANOM_TENSORRT_ROOT)
        list(APPEND _anom_tensorrt_include_hints "${ANOM_TENSORRT_ROOT}/include")
        list(APPEND _anom_tensorrt_library_hints
            "${ANOM_TENSORRT_ROOT}/lib" "${ANOM_TENSORRT_ROOT}/lib64")
    endif()
    find_path(ANOM_TENSORRT_INCLUDE_DIR NAMES NvInfer.h
        HINTS ${_anom_tensorrt_include_hints})
    find_library(ANOM_TENSORRT_NVINFER_LIBRARY NAMES nvinfer_10 nvinfer
        HINTS ${_anom_tensorrt_library_hints})
    find_library(ANOM_TENSORRT_PLUGIN_LIBRARY NAMES nvinfer_plugin_10 nvinfer_plugin
        HINTS ${_anom_tensorrt_library_hints})
    find_library(ANOM_TENSORRT_ONNX_PARSER_LIBRARY NAMES nvonnxparser_10 nvonnxparser
        HINTS ${_anom_tensorrt_library_hints})
    if(NOT ANOM_TENSORRT_INCLUDE_DIR OR
       NOT ANOM_TENSORRT_NVINFER_LIBRARY OR
       NOT ANOM_TENSORRT_PLUGIN_LIBRARY OR
       NOT ANOM_TENSORRT_ONNX_PARSER_LIBRARY)
        message(FATAL_ERROR
            "TensorRT was requested but was not found. Set ANOM_TENSORRT_ROOT to an absolute SDK path.")
    endif()
    find_package(CUDAToolkit REQUIRED)
    foreach(_anom_trt_component IN ITEMS nvinfer plugin onnx_parser)
        add_library(anom_tensorrt_${_anom_trt_component} UNKNOWN IMPORTED GLOBAL)
        add_library(anom::tensorrt_${_anom_trt_component}
            ALIAS anom_tensorrt_${_anom_trt_component})
    endforeach()
    set_target_properties(anom_tensorrt_nvinfer PROPERTIES
        IMPORTED_LOCATION "${ANOM_TENSORRT_NVINFER_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${ANOM_TENSORRT_INCLUDE_DIR}")
    set_target_properties(anom_tensorrt_plugin PROPERTIES
        IMPORTED_LOCATION "${ANOM_TENSORRT_PLUGIN_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${ANOM_TENSORRT_INCLUDE_DIR}")
    set_target_properties(anom_tensorrt_onnx_parser PROPERTIES
        IMPORTED_LOCATION "${ANOM_TENSORRT_ONNX_PARSER_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${ANOM_TENSORRT_INCLUDE_DIR}")
endif()

if(ANOM_ENABLE_ONNXRUNTIME)
    set(_anom_ort_include_hints "")
    set(_anom_ort_library_hints "")
    if(ANOM_ONNXRUNTIME_ROOT)
        list(APPEND _anom_ort_include_hints
            "${ANOM_ONNXRUNTIME_ROOT}/include"
            "${ANOM_ONNXRUNTIME_ROOT}/build/native/include")
        list(APPEND _anom_ort_library_hints
            "${ANOM_ONNXRUNTIME_ROOT}/lib"
            "${ANOM_ONNXRUNTIME_ROOT}/lib64"
            "${ANOM_ONNXRUNTIME_ROOT}/build/native/lib/x64"
            "${ANOM_ONNXRUNTIME_ROOT}/runtimes/win-x64/native")
    endif()
    find_path(ANOM_ONNXRUNTIME_INCLUDE_DIR NAMES onnxruntime_cxx_api.h
        HINTS ${_anom_ort_include_hints})
    find_library(ANOM_ONNXRUNTIME_LIBRARY NAMES onnxruntime
        HINTS ${_anom_ort_library_hints})
    if(NOT ANOM_ONNXRUNTIME_INCLUDE_DIR OR NOT ANOM_ONNXRUNTIME_LIBRARY)
        message(FATAL_ERROR
            "ONNX Runtime was requested but was not found. Set ANOM_ONNXRUNTIME_ROOT to an absolute SDK path.")
    endif()

    if(WIN32)
        set(_anom_ort_runtime_hints "")
        if(ANOM_ONNXRUNTIME_ROOT)
            list(APPEND _anom_ort_runtime_hints
                "${ANOM_ONNXRUNTIME_ROOT}/bin"
                "${ANOM_ONNXRUNTIME_ROOT}/lib"
                "${ANOM_ONNXRUNTIME_ROOT}/runtimes/win-x64/native")
        endif()
        find_file(ANOM_ONNXRUNTIME_SHARED_LIBRARY NAMES onnxruntime.dll
            HINTS ${_anom_ort_runtime_hints})
        if(NOT ANOM_ONNXRUNTIME_SHARED_LIBRARY)
            message(FATAL_ERROR "onnxruntime.dll was not found")
        endif()
        add_library(anom_onnxruntime SHARED IMPORTED GLOBAL)
        set_target_properties(anom_onnxruntime PROPERTIES
            IMPORTED_IMPLIB "${ANOM_ONNXRUNTIME_LIBRARY}"
            IMPORTED_LOCATION "${ANOM_ONNXRUNTIME_SHARED_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${ANOM_ONNXRUNTIME_INCLUDE_DIR}")
    else()
        add_library(anom_onnxruntime UNKNOWN IMPORTED GLOBAL)
        set_target_properties(anom_onnxruntime PROPERTIES
            IMPORTED_LOCATION "${ANOM_ONNXRUNTIME_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${ANOM_ONNXRUNTIME_INCLUDE_DIR}")
    endif()
    add_library(anom::onnxruntime ALIAS anom_onnxruntime)

endif()

# Core library and public C API -------------------------------------------------

set(ANOM_CORE_SOURCES
    "${ANOM_PRIVATE_SOURCE_ROOT}/adapters/adapter.cpp"
    "${ANOM_PRIVATE_SOURCE_ROOT}/backends/backend_factory.cpp"
    "${ANOM_PRIVATE_SOURCE_ROOT}/infrastructure/json.cpp"
    "${ANOM_PRIVATE_SOURCE_ROOT}/infrastructure/sha256.cpp"
    "${ANOM_PRIVATE_SOURCE_ROOT}/model/inference_session.cpp"
    "${ANOM_PRIVATE_SOURCE_ROOT}/model/manifest.cpp"
    "${ANOM_PRIVATE_SOURCE_ROOT}/model/types.cpp"
    "${ANOM_PRIVATE_SOURCE_ROOT}/plugins/dynamic_library.cpp"
    "${ANOM_PRIVATE_SOURCE_ROOT}/processing/analyzer.cpp"
    "${ANOM_PRIVATE_SOURCE_ROOT}/processing/postprocessor.cpp"
    "${ANOM_PRIVATE_SOURCE_ROOT}/processing/preprocessor.cpp")

add_library(anom_internal STATIC)
add_library(anom::model ALIAS anom_internal)
target_sources(anom_internal PRIVATE ${ANOM_CORE_SOURCES})
target_include_directories(anom_internal
    PUBLIC "$<BUILD_INTERFACE:${ANOM_PRIVATE_SOURCE_ROOT}>")
target_link_libraries(anom_internal PUBLIC anom::opencv)
if(CMAKE_DL_LIBS)
    target_link_libraries(anom_internal PUBLIC "${CMAKE_DL_LIBS}")
endif()
anom_apply_project_defaults(anom_internal)

add_library(anom_engine SHARED)
add_library(anom::engine ALIAS anom_engine)
target_sources(anom_engine PRIVATE
    "${ANOM_PRIVATE_SOURCE_ROOT}/c_api/anom_engine.cpp"
    "${ANOM_PRIVATE_SOURCE_ROOT}/c_api/algorithm_objects.cpp"
    "${ANOM_PRIVATE_SOURCE_ROOT}/c_api/model_management.cpp")
if(ANOM_BUILD_ALGO_PATCHCORE OR ANOM_BUILD_ALGO_PADIM)
    target_sources(anom_engine PRIVATE
        "${ANOM_PRIVATE_SOURCE_ROOT}/adapters/feature_utils.cpp")
endif()
if(ANOM_BUILD_ALGO_PATCHCORE)
    target_sources(anom_engine PRIVATE
        "${ANOM_PRIVATE_SOURCE_ROOT}/pipeline/patchcore.cpp")
    target_link_libraries(anom_engine PRIVATE anom::faiss anom::faiss_c)
endif()
if(ANOM_BUILD_ALGO_PADIM)
    target_sources(anom_engine PRIVATE
        "${ANOM_PRIVATE_SOURCE_ROOT}/pipeline/padim.cpp")
endif()
target_compile_definitions(anom_engine PRIVATE
    ANOM_ENGINE_BUILD=1 ANOM_ENGINE_VERSION_STRING="${PROJECT_VERSION}"
    ANOM_ENGINE_HAS_PATCHCORE=$<BOOL:${ANOM_BUILD_ALGO_PATCHCORE}>
    ANOM_ENGINE_HAS_PADIM=$<BOOL:${ANOM_BUILD_ALGO_PADIM}>
    ANOM_ENGINE_HAS_DIRECT=$<BOOL:${ANOM_BUILD_ALGO_DIRECT}>
    ANOM_ENGINE_HAS_EFFICIENTAD=$<BOOL:${ANOM_BUILD_ALGO_EFFICIENTAD}>
    ANOM_ENGINE_HAS_DFKDE=$<BOOL:${ANOM_BUILD_ALGO_DFKDE}>
    ANOM_ENGINE_HAS_SPADE=$<BOOL:${ANOM_BUILD_ALGO_SPADE}>
    ANOM_ENGINE_HAS_YOLO=$<BOOL:${ANOM_BUILD_ALGO_YOLO}>)
target_include_directories(anom_engine
    PUBLIC
        "$<BUILD_INTERFACE:${ANOM_PUBLIC_INCLUDE_ROOT}>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
    PRIVATE "${ANOM_PRIVATE_SOURCE_ROOT}")
target_link_libraries(anom_engine PRIVATE anom_internal)
set_target_properties(anom_engine PROPERTIES
    OUTPUT_NAME "anomEngine"
    VERSION "${PROJECT_VERSION}"
    SOVERSION "3"
    INSTALL_RPATH "${ANOM_ORIGIN_RPATH}")
anom_apply_project_defaults(anom_engine)

# Dynamically loaded algorithm plugins -----------------------------------------

set(ANOM_FAISS_SEARCH_SOURCES
    "${ANOM_PRIVATE_SOURCE_ROOT}/retrieval/faiss_index.cpp"
    "${ANOM_PRIVATE_SOURCE_ROOT}/retrieval/gpu_faiss_client.cpp"
    "${ANOM_PRIVATE_SOURCE_ROOT}/plugins/dynamic_library.cpp")

set(ANOM_PLUGIN_CONTRACT_SOURCES
    "${ANOM_PRIVATE_SOURCE_ROOT}/infrastructure/json.cpp"
    "${ANOM_PRIVATE_SOURCE_ROOT}/infrastructure/sha256.cpp"
    "${ANOM_PRIVATE_SOURCE_ROOT}/model/manifest.cpp"
    "${ANOM_PRIVATE_SOURCE_ROOT}/model/types.cpp")

set(ANOM_ALGORITHM_TARGETS "")
function(anom_add_algorithm name definition adapter_source)
    set(_anom_target "anom_algo_${name}")
    add_library(${_anom_target} MODULE)
    target_sources(${_anom_target} PRIVATE
        "${ANOM_PRIVATE_SOURCE_ROOT}/plugins/algorithm_plugin.cpp"
        "${adapter_source}" ${ANOM_PLUGIN_CONTRACT_SOURCES} ${ARGN})
    target_compile_definitions(${_anom_target} PRIVATE
        ${definition}=1 ANOM_ALGORITHM_NAME="${name}")
    target_include_directories(${_anom_target} PRIVATE "${ANOM_PRIVATE_SOURCE_ROOT}")
    target_link_libraries(${_anom_target} PRIVATE anom::opencv)
    set_target_properties(${_anom_target} PROPERTIES
        PREFIX "" OUTPUT_NAME "${_anom_target}"
        INSTALL_RPATH "${ANOM_ORIGIN_RPATH}")
    anom_apply_project_defaults(${_anom_target})
    set(ANOM_ALGORITHM_TARGETS ${ANOM_ALGORITHM_TARGETS} ${_anom_target} PARENT_SCOPE)
endfunction()

if(ANOM_BUILD_ALGO_DIRECT)
    anom_add_algorithm(direct ANOM_ALGORITHM_DIRECT
        "${ANOM_PRIVATE_SOURCE_ROOT}/adapters/direct_adapter.cpp")
endif()
if(ANOM_BUILD_ALGO_EFFICIENTAD)
    anom_add_algorithm(efficientad ANOM_ALGORITHM_EFFICIENTAD
        "${ANOM_PRIVATE_SOURCE_ROOT}/adapters/efficientad_adapter.cpp"
        "${ANOM_PRIVATE_SOURCE_ROOT}/adapters/feature_utils.cpp")
endif()
if(ANOM_BUILD_ALGO_DFKDE)
    anom_add_algorithm(dfkde ANOM_ALGORITHM_DFKDE
        "${ANOM_PRIVATE_SOURCE_ROOT}/adapters/dfkde_adapter.cpp"
        "${ANOM_PRIVATE_SOURCE_ROOT}/adapters/feature_utils.cpp")
endif()
if(ANOM_BUILD_ALGO_PADIM)
    anom_add_algorithm(padim ANOM_ALGORITHM_PADIM
        "${ANOM_PRIVATE_SOURCE_ROOT}/adapters/padim_adapter.cpp"
        "${ANOM_PRIVATE_SOURCE_ROOT}/adapters/feature_utils.cpp")
endif()
if(ANOM_BUILD_ALGO_PATCHCORE)
    anom_add_algorithm(patchcore ANOM_ALGORITHM_PATCHCORE
        "${ANOM_PRIVATE_SOURCE_ROOT}/adapters/patchcore_adapter.cpp"
        "${ANOM_PRIVATE_SOURCE_ROOT}/adapters/feature_utils.cpp" ${ANOM_FAISS_SEARCH_SOURCES})
    target_link_libraries(anom_algo_patchcore PRIVATE anom::faiss anom::faiss_c ${CMAKE_DL_LIBS})
endif()
if(ANOM_BUILD_ALGO_SPADE)
    anom_add_algorithm(spade ANOM_ALGORITHM_SPADE
        "${ANOM_PRIVATE_SOURCE_ROOT}/adapters/spade_adapter.cpp"
        "${ANOM_PRIVATE_SOURCE_ROOT}/adapters/feature_utils.cpp" ${ANOM_FAISS_SEARCH_SOURCES})
    target_link_libraries(anom_algo_spade PRIVATE anom::faiss anom::faiss_c ${CMAKE_DL_LIBS})
endif()
if(ANOM_BUILD_ALGO_YOLO)
    anom_add_algorithm(yolo ANOM_ALGORITHM_YOLO
        "${ANOM_PRIVATE_SOURCE_ROOT}/adapters/yolo_adapter.cpp")
endif()

# Dynamically loaded backend plugins -------------------------------------------

set(ANOM_BACKEND_TARGETS "")
if(ANOM_ENABLE_TENSORRT)
    add_library(anom_backend_tensorrt MODULE)
    target_sources(anom_backend_tensorrt PRIVATE
        "${ANOM_PRIVATE_SOURCE_ROOT}/plugins/backend_plugin.cpp"
        "${ANOM_PRIVATE_SOURCE_ROOT}/backends/tensorrt_backend.cpp"
        "${ANOM_PRIVATE_SOURCE_ROOT}/model/types.cpp")
    target_compile_definitions(anom_backend_tensorrt PRIVATE
        ANOM_BACKEND_TENSORRT=1 ANOM_BACKEND_NAME="tensorrt")
    target_include_directories(anom_backend_tensorrt PRIVATE "${ANOM_PRIVATE_SOURCE_ROOT}")
    target_link_libraries(anom_backend_tensorrt PRIVATE
        anom::opencv anom::tensorrt_nvinfer anom::tensorrt_plugin
        anom::tensorrt_onnx_parser CUDA::cudart)
    set_target_properties(anom_backend_tensorrt PROPERTIES
        PREFIX "" OUTPUT_NAME "anom_backend_tensorrt"
        INSTALL_RPATH "${ANOM_ORIGIN_RPATH}")
    anom_apply_project_defaults(anom_backend_tensorrt)
    list(APPEND ANOM_BACKEND_TARGETS anom_backend_tensorrt)
endif()

if(ANOM_ENABLE_ONNXRUNTIME)
    add_library(anom_backend_onnxruntime MODULE)
    target_sources(anom_backend_onnxruntime PRIVATE
        "${ANOM_PRIVATE_SOURCE_ROOT}/plugins/backend_plugin.cpp"
        "${ANOM_PRIVATE_SOURCE_ROOT}/backends/ort_backend.cpp"
        "${ANOM_PRIVATE_SOURCE_ROOT}/model/types.cpp")
    target_compile_definitions(anom_backend_onnxruntime PRIVATE
        ANOM_BACKEND_ONNXRUNTIME=1 ANOM_BACKEND_NAME="onnxruntime")
    target_include_directories(anom_backend_onnxruntime PRIVATE "${ANOM_PRIVATE_SOURCE_ROOT}")
    target_link_libraries(anom_backend_onnxruntime PRIVATE anom::opencv anom::onnxruntime)
    set_target_properties(anom_backend_onnxruntime PROPERTIES
        PREFIX "" OUTPUT_NAME "anom_backend_onnxruntime"
        INSTALL_RPATH "${ANOM_ORIGIN_RPATH}")
    anom_apply_project_defaults(anom_backend_onnxruntime)
    list(APPEND ANOM_BACKEND_TARGETS anom_backend_onnxruntime)
endif()

# Tests ------------------------------------------------------------------------

set(ANOM_SEARCH_TARGETS "")
if(ANOM_ENABLE_FAISS_GPU)
    # A separate static GPU SDK avoids loading a CUDA-dependent faiss.dll into
    # the CPU algorithm plugins (or colliding with their CPU faiss.dll).
    set(ANOM_FAISS_GPU_ROOT "${ANOM_BUNDLED_DEPENDENCY_ROOT}/faiss-gpu" CACHE PATH
        "Static GPU-enabled Faiss SDK, separate from the CPU Faiss distribution")
    anom_assert_absolute_path(ANOM_FAISS_GPU_ROOT)
    find_path(ANOM_FAISS_GPU_INCLUDE_DIR faiss/gpu/StandardGpuResources.h
        PATHS "${ANOM_FAISS_GPU_ROOT}/include" NO_DEFAULT_PATH)
    find_file(ANOM_FAISS_GPU_LIBRARY NAMES faiss.lib libfaiss.a
        PATHS "${ANOM_FAISS_GPU_ROOT}/lib" "${ANOM_FAISS_GPU_ROOT}/lib64" NO_DEFAULT_PATH)
    if(NOT ANOM_FAISS_GPU_INCLUDE_DIR OR NOT ANOM_FAISS_GPU_LIBRARY)
        message(FATAL_ERROR "GPU Faiss requires a static GPU-enabled SDK. Set ANOM_FAISS_GPU_ROOT (CPU Faiss is insufficient).")
    endif()
    if(WIN32)
        # An import library would pull a second faiss.dll into the process and
        # defeat CPU/GPU dependency isolation. Reject it even if the DLL itself
        # was omitted from the SDK directory.
        file(STRINGS "${ANOM_FAISS_GPU_LIBRARY}" _anom_faiss_gpu_imports REGEX "[Ff][Aa][Ii][Ss][Ss].*\\.[Dd][Ll][Ll]")
        if(_anom_faiss_gpu_imports)
            message(FATAL_ERROR "ANOM_FAISS_GPU_LIBRARY must be a static library, not a Faiss DLL import library.")
        endif()
    endif()
    find_package(CUDAToolkit REQUIRED)
    find_package(OpenMP REQUIRED COMPONENTS CXX)
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
    add_library(anom_search_faiss_cuda MODULE "${ANOM_PRIVATE_SOURCE_ROOT}/plugins/faiss_cuda_plugin.cpp")
    target_include_directories(anom_search_faiss_cuda PRIVATE "${ANOM_PRIVATE_SOURCE_ROOT}")
    target_include_directories(anom_search_faiss_cuda SYSTEM PRIVATE "${ANOM_FAISS_GPU_INCLUDE_DIR}")
    target_link_libraries(anom_search_faiss_cuda PRIVATE "${ANOM_FAISS_GPU_LIBRARY}"
        CUDA::cudart CUDA::cublas OpenMP::OpenMP_CXX ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
    if(UNIX AND NOT APPLE)
        target_link_options(anom_search_faiss_cuda PRIVATE "-Wl,-Bsymbolic" "-Wl,--exclude-libs,ALL")
    endif()
    set_target_properties(anom_search_faiss_cuda PROPERTIES PREFIX "" INSTALL_RPATH "${ANOM_ORIGIN_RPATH}")
    anom_apply_project_defaults(anom_search_faiss_cuda)
    list(APPEND ANOM_SEARCH_TARGETS anom_search_faiss_cuda)
endif()

if(BUILD_TESTING)
    add_subdirectory(tests/object_api)

    add_library(c_header_compile_test OBJECT "${ANOM_TEST_ROOT}/c_header_compile_test.c")
    target_compile_features(c_header_compile_test PRIVATE c_std_11)
    set_target_properties(c_header_compile_test PROPERTIES C_EXTENSIONS NO)
    target_include_directories(c_header_compile_test PRIVATE "${ANOM_PUBLIC_INCLUDE_ROOT}")
    if(MSVC)
        target_compile_options(c_header_compile_test PRIVATE /W4 /utf-8)
    elseif(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(c_header_compile_test PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    add_library(anom_training_test_support STATIC)
    target_sources(anom_training_test_support PRIVATE
        "${ANOM_PRIVATE_SOURCE_ROOT}/adapters/feature_utils.cpp"
        "${ANOM_PRIVATE_SOURCE_ROOT}/pipeline/dfkde.cpp"
        "${ANOM_PRIVATE_SOURCE_ROOT}/pipeline/efficientad.cpp"
        "${ANOM_PRIVATE_SOURCE_ROOT}/pipeline/padim.cpp"
        "${ANOM_PRIVATE_SOURCE_ROOT}/pipeline/patchcore.cpp"
        "${ANOM_PRIVATE_SOURCE_ROOT}/pipeline/spade.cpp")
    target_include_directories(anom_training_test_support PUBLIC "${ANOM_PRIVATE_SOURCE_ROOT}")
    target_link_libraries(anom_training_test_support PUBLIC
        anom_internal anom::opencv anom::faiss anom::faiss_c)
    anom_apply_project_defaults(anom_training_test_support)

    add_executable(model_contract_tests "${ANOM_TEST_ROOT}/model_contract_tests.cpp")
    target_link_libraries(model_contract_tests PRIVATE anom_training_test_support)
    target_compile_definitions(model_contract_tests PRIVATE ANOM_SOURCE_DIR="${ANOM_SOURCE_ROOT}")
    if(ANOM_ALGORITHM_TARGETS)
        add_dependencies(model_contract_tests ${ANOM_ALGORITHM_TARGETS})
    endif()
    anom_apply_project_defaults(model_contract_tests)
    add_test(NAME model_contract_tests COMMAND model_contract_tests)

    if(ANOM_ENABLE_ONNXRUNTIME AND ANOM_BUILD_ALGO_DIRECT)
        add_library(legacy_backend_plugin MODULE "${ANOM_TEST_ROOT}/legacy_backend_plugin.cpp")
        target_include_directories(legacy_backend_plugin PRIVATE "${ANOM_PRIVATE_SOURCE_ROOT}")
        set_target_properties(legacy_backend_plugin PROPERTIES
            PREFIX "" OUTPUT_NAME "anom_backend_onnxruntime"
            ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/legacy_plugins"
            LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/legacy_plugins"
            RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/legacy_plugins")
        anom_apply_project_defaults(legacy_backend_plugin)
        add_executable(ort_backend_tests "${ANOM_TEST_ROOT}/ort_backend_tests.cpp")
        target_link_libraries(ort_backend_tests PRIVATE anom_internal)
        target_compile_definitions(ort_backend_tests PRIVATE
            ANOM_TEST_DATA_DIR="${ANOM_TEST_ROOT}/data"
            ANOM_TEST_LEGACY_PLUGIN_DIR="$<TARGET_FILE_DIR:legacy_backend_plugin>")
        add_dependencies(ort_backend_tests anom_backend_onnxruntime anom_algo_direct legacy_backend_plugin)
        anom_apply_project_defaults(ort_backend_tests)
        add_test(NAME ort_backend_tests COMMAND ort_backend_tests)

        if(ANOM_ENABLE_TENSORRT)
            add_executable(runtime_device_tests "${ANOM_TEST_ROOT}/runtime_device_tests.cpp")
            target_link_libraries(runtime_device_tests PRIVATE anom_internal)
            target_compile_definitions(runtime_device_tests PRIVATE ANOM_TEST_DATA_DIR="${ANOM_TEST_ROOT}/data")
            add_dependencies(runtime_device_tests anom_backend_tensorrt anom_backend_onnxruntime)
            anom_apply_project_defaults(runtime_device_tests)
            add_test(NAME tensorrt_device_tests COMMAND runtime_device_tests)
            set_tests_properties(tensorrt_device_tests PROPERTIES SKIP_RETURN_CODE 77)
        endif()

        add_executable(c_api_contract_tests "${ANOM_TEST_ROOT}/c_api_contract_tests.cpp")
        target_include_directories(c_api_contract_tests PRIVATE "${ANOM_PRIVATE_SOURCE_ROOT}")
        target_link_libraries(c_api_contract_tests PRIVATE anom::engine)
        target_compile_definitions(c_api_contract_tests PRIVATE
            ANOM_TEST_DATA_DIR="${ANOM_TEST_ROOT}/data"
            ANOM_TEST_TENSORRT_ENABLED=$<BOOL:${ANOM_ENABLE_TENSORRT}>)
        add_dependencies(c_api_contract_tests anom_backend_onnxruntime anom_algo_direct)
        if(ANOM_BUILD_ALGO_PATCHCORE)
            add_dependencies(c_api_contract_tests anom_algo_patchcore)
        endif()
        anom_apply_project_defaults(c_api_contract_tests)
        add_test(NAME c_api_contract_tests COMMAND c_api_contract_tests)
    endif()
endif()

if(BUILD_TESTING)
    add_subdirectory(tests/backend_memory)
    add_subdirectory(tests/faiss_runtime)
endif()

# Installation and Windows runtime deployment ---------------------------------

install(TARGETS anom_engine
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")

set(ANOM_PLUGIN_TARGETS ${ANOM_ALGORITHM_TARGETS} ${ANOM_BACKEND_TARGETS} ${ANOM_SEARCH_TARGETS})
if(ANOM_PLUGIN_TARGETS)
    if(WIN32)
        set(ANOM_PLUGIN_INSTALL_DIRECTORY "${CMAKE_INSTALL_BINDIR}")
    else()
        set(ANOM_PLUGIN_INSTALL_DIRECTORY "${CMAKE_INSTALL_LIBDIR}")
    endif()
    install(TARGETS ${ANOM_PLUGIN_TARGETS}
        RUNTIME DESTINATION "${ANOM_PLUGIN_INSTALL_DIRECTORY}"
        LIBRARY DESTINATION "${ANOM_PLUGIN_INSTALL_DIRECTORY}")
endif()
install(FILES
    "${ANOM_PUBLIC_INCLUDE_ROOT}/anomEngine/anomEngine.h"
    "${ANOM_PUBLIC_INCLUDE_ROOT}/anomEngine/common.h"
    "${ANOM_PUBLIC_INCLUDE_ROOT}/anomEngine/direct.h"
    "${ANOM_PUBLIC_INCLUDE_ROOT}/anomEngine/efficientad.h"
    "${ANOM_PUBLIC_INCLUDE_ROOT}/anomEngine/dfkde.h"
    "${ANOM_PUBLIC_INCLUDE_ROOT}/anomEngine/padim.h"
    "${ANOM_PUBLIC_INCLUDE_ROOT}/anomEngine/patchcore.h"
    "${ANOM_PUBLIC_INCLUDE_ROOT}/anomEngine/spade.h"
    "${ANOM_PUBLIC_INCLUDE_ROOT}/anomEngine/yolo.h"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/anomEngine")

if(WIN32)
    set(ANOM_RUNTIME_FILES "")
    if(ANOM_OPENCV_ROOT)
        string(REPLACE "." "" _anom_opencv_version_compact "${OpenCV_VERSION}")
        set(ANOM_OPENCV_RUNTIME_ROOT "${ANOM_OPENCV_ROOT}/x64/vc16/bin"
            CACHE PATH "Absolute directory containing the OpenCV runtime DLLs")
        anom_assert_absolute_path(ANOM_OPENCV_RUNTIME_ROOT)
        list(APPEND ANOM_RUNTIME_FILES
            "$<IF:$<CONFIG:Debug>,${ANOM_OPENCV_RUNTIME_ROOT}/opencv_world${_anom_opencv_version_compact}d.dll,${ANOM_OPENCV_RUNTIME_ROOT}/opencv_world${_anom_opencv_version_compact}.dll>")
    endif()
    if(ANOM_ENABLE_TENSORRT AND ANOM_TENSORRT_ROOT)
        file(GLOB _anom_tensorrt_runtime_files CONFIGURE_DEPENDS
            "${ANOM_TENSORRT_ROOT}/bin/*.dll")
        list(APPEND ANOM_RUNTIME_FILES ${_anom_tensorrt_runtime_files})
    endif()
    if(ANOM_NEEDS_FAISS AND ANOM_FAISS_ROOT)
        foreach(_anom_faiss_runtime_name IN ITEMS
                faiss.dll faiss_c.dll libomp.dll libopenblas.dll)
            set(_anom_faiss_runtime_file "${ANOM_FAISS_ROOT}/bin/${_anom_faiss_runtime_name}")
            if(EXISTS "${_anom_faiss_runtime_file}")
                list(APPEND ANOM_RUNTIME_FILES "${_anom_faiss_runtime_file}")
            endif()
        endforeach()
    endif()
    if(ANOM_ENABLE_ONNXRUNTIME)
        list(APPEND ANOM_RUNTIME_FILES
            "$<TARGET_FILE:anom::onnxruntime>")
    endif()

    function(anom_deploy_runtime_dependencies target_name)
        if(ANOM_RUNTIME_FILES)
            add_custom_command(TARGET ${target_name} POST_BUILD
                COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                    ${ANOM_RUNTIME_FILES} "$<TARGET_FILE_DIR:${target_name}>"
                COMMENT "Deploying runtime dependencies for ${target_name}"
                COMMAND_EXPAND_LISTS VERBATIM)
        endif()
    endfunction()

    anom_deploy_runtime_dependencies(anom_engine)
    if(ANOM_RUNTIME_FILES)
        install(FILES ${ANOM_RUNTIME_FILES} DESTINATION "${CMAKE_INSTALL_BINDIR}")
    endif()
endif()
