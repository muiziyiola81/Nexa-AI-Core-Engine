cmake_minimum_required(VERSION 3.16)
project(nexa_native_runtime VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# User-configurable SIMD optimization toggles (Architecture Agnostic)
option(ENABLE_AVX2 "Enable AVX2 and FMA SIMD optimizations on x86_64" ON)
option(ENABLE_ARM_NEON "Enable ARM NEON SIMD optimizations on ARM/ARM64" ON)

include(CheckCXXCompilerFlag)

# Architecture Detection and Conditional Flag Configuration
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    message(STATUS "[NEXA CMAKE] Detected ARM64 architecture (${CMAKE_SYSTEM_PROCESSOR})")
    if(ENABLE_ARM_NEON)
        check_cxx_compiler_flag("-march=armv8-a+simd" COMPILER_SUPPORTS_ARMV8_SIMD)
        if(COMPILER_SUPPORTS_ARMV8_SIMD)
            add_compile_options(-march=armv8-a+simd)
            message(STATUS "[NEXA CMAKE] Enabled ARM64 NEON SIMD (-march=armv8-a+simd)")
        endif()
    endif()
    add_compile_options(-O3 -pthread)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    message(STATUS "[NEXA CMAKE] Detected x86_64 architecture (${CMAKE_SYSTEM_PROCESSOR})")
    if(ENABLE_AVX2)
        check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
        check_cxx_compiler_flag("-mfma" COMPILER_SUPPORTS_FMA)
        if(COMPILER_SUPPORTS_AVX2 AND COMPILER_SUPPORTS_FMA)
            add_compile_options(-mavx2 -mfma)
            message(STATUS "[NEXA CMAKE] Enabled x86_64 AVX2 & FMA SIMD (-mavx2 -mfma)")
        else()
            message(STATUS "[NEXA CMAKE] Host or compiler lacks AVX2/FMA; continuing with standard -O3")
        endif()
    else()
        message(STATUS "[NEXA CMAKE] x86_64 AVX2/FMA manually disabled via ENABLE_AVX2=OFF")
    endif()
    add_compile_options(-O3 -pthread)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "armv7l|armv8l|armhf")
    message(WARNING "[NEXA CMAKE] Detected 32-bit ARM (${CMAKE_SYSTEM_PROCESSOR}). 32-bit memory address space is restricted to < 2GB.")
    if(ENABLE_ARM_NEON)
        check_cxx_compiler_flag("-mfpu=neon -mfloat-abi=hard" COMPILER_SUPPORTS_NEON_32)
        if(COMPILER_SUPPORTS_NEON_32)
            add_compile_options(-mfpu=neon -mfloat-abi=hard)
        endif()
    endif()
    add_compile_options(-O3 -pthread)
else()
    message(STATUS "[NEXA CMAKE] Generic architecture (${CMAKE_SYSTEM_PROCESSOR}). Standard -O3 optimization enabled.")
    add_compile_options(-O3 -pthread)
endif()

include_directories(include)

# Core Library Sources
set(NEXA_CORE_SRCS
    src/tensor.cpp
    src/nn.cpp
    src/tokenizer.cpp
    src/model_loader.cpp
    src/autograd.cpp
    src/optimizer.cpp
    src/nano_model.cpp
    src/dataset.cpp
    src/trainer.cpp
    src/experiment.cpp
)

# Shared / Static Library
add_library(nexa_core STATIC ${NEXA_CORE_SRCS})

# Standalone CLI Executable
add_executable(nexa-runtime-cli src/main.cpp)
target_link_libraries(nexa-runtime-cli PRIVATE nexa_core)

# Install Target
install(TARGETS nexa-runtime-cli nexa_core
    RUNTIME DESTINATION bin
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)
