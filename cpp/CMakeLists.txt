cmake_minimum_required(VERSION 3.20)
project(pianofi_live CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(PF_USE_SPSC "Use the lock-free SpscRingBuffer" ON)
option(PF_WITH_ONNX "Build the engine (needs ONNX Runtime)" ON)
set(ONNXRUNTIME_DIR "" CACHE PATH "Unzipped ONNX Runtime release (has include/ and lib/)")

if(WIN32)
  add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
endif()
if(MSVC)
  add_compile_options(/W4 /permissive- /utf-8)
else()
  add_compile_options(-Wall -Wextra)
endif()

find_package(Threads REQUIRED)

add_library(pf_core STATIC src/wav.cpp src/mel.cpp src/decoder.cpp src/capture.cpp)
target_include_directories(pf_core PUBLIC include PRIVATE third_party)
target_link_libraries(pf_core PUBLIC Threads::Threads)
if(PF_USE_SPSC)
  target_compile_definitions(pf_core PUBLIC PF_USE_SPSC)
  target_sources(pf_core PRIVATE src/spsc_ring_buffer.cpp)
endif()

add_executable(pf_tests tests/test_main.cpp)
target_link_libraries(pf_tests PRIVATE pf_core)

add_executable(bench_ring tools/bench_ring.cpp)
target_link_libraries(bench_ring PRIVATE pf_core)

add_executable(mel_dump tools/mel_dump.cpp)
target_link_libraries(mel_dump PRIVATE pf_core)

if(PF_WITH_ONNX)
  if(NOT ONNXRUNTIME_DIR)
    message(FATAL_ERROR "Set -DONNXRUNTIME_DIR=<path to unzipped onnxruntime release>, or -DPF_WITH_ONNX=OFF")
  endif()
  find_library(ORT_LIB onnxruntime PATHS ${ONNXRUNTIME_DIR}/lib NO_DEFAULT_PATH REQUIRED)
  add_executable(pianofi_live src/engine.cpp src/model.cpp src/main.cpp)
  target_include_directories(pianofi_live PRIVATE ${ONNXRUNTIME_DIR}/include)
  target_link_libraries(pianofi_live PRIVATE pf_core ${ORT_LIB})
  if(WIN32)
    # Copy the ONNX Runtime DLLs next to the exe so it runs without PATH edits.
    file(GLOB ORT_DLLS ${ONNXRUNTIME_DIR}/lib/*.dll)
    foreach(dll ${ORT_DLLS})
      add_custom_command(TARGET pianofi_live POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${dll} $<TARGET_FILE_DIR:pianofi_live>)
    endforeach()
  endif()
endif()

add_executable(decode_dump tools/decode_dump.cpp)
target_link_libraries(decode_dump PRIVATE pf_core)
