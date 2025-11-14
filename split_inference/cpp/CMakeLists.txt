cmake_minimum_required(VERSION 3.20)
project(split_inference_scheduler LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(ENABLE_SYCL "Enable SYCL/DPC++ support for iGPU" OFF)
option(ENABLE_ONEDNN "Enable oneDNN for CPU kernels" OFF)
option(ENABLE_VTUNE "Enable Intel VTune profiling" OFF)

# Find required packages
find_package(PkgConfig REQUIRED)

# ZeroMQ
pkg_check_modules(ZMQ REQUIRED libzmq)
include_directories(${ZMQ_INCLUDE_DIRS})
link_directories(${ZMQ_LIBRARY_DIRS})

# JSON library (header-only)
find_path(NLOHMANN_JSON_INCLUDE_DIR nlohmann/json.hpp
    PATHS /usr/include /usr/local/include
    PATH_SUFFIXES nlohmann
)
if(NOT NLOHMANN_JSON_INCLUDE_DIR)
    message(STATUS "nlohmann/json not found, will download")
    include(FetchContent)
    FetchContent_Declare(json
        URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz
    )
    FetchContent_MakeAvailable(json)
else()
    message(STATUS "Found nlohmann/json: ${NLOHMANN_JSON_INCLUDE_DIR}")
    include_directories(${NLOHMANN_JSON_INCLUDE_DIR})
endif()

# SYCL/DPC++ support
if(ENABLE_SYCL)
    message(STATUS "SYCL support enabled")
    
    # Try to find Intel oneAPI
    if(DEFINED ENV{ONEAPI_ROOT})
        set(ONEAPI_ROOT $ENV{ONEAPI_ROOT})
        message(STATUS "oneAPI root: ${ONEAPI_ROOT}")
        
        # Add SYCL compiler flags
        set(CMAKE_CXX_COMPILER ${ONEAPI_ROOT}/compiler/latest/linux/bin/icpx)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")
        
        add_definitions(-DENABLE_SYCL)
    else()
        message(WARNING "ONEAPI_ROOT not set, SYCL features will be limited")
    endif()
endif()

# oneDNN support
if(ENABLE_ONEDNN)
    message(STATUS "oneDNN support enabled")
    find_package(dnnl REQUIRED)
    add_definitions(-DENABLE_ONEDNN)
endif()

# VTune support
if(ENABLE_VTUNE)
    message(STATUS "VTune profiling enabled")
    find_path(VTUNE_INCLUDE_DIR ittnotify.h
        PATHS ENV VTUNE_PROFILER_DIR
        PATH_SUFFIXES include
    )
    if(VTUNE_INCLUDE_DIR)
        include_directories(${VTUNE_INCLUDE_DIR})
        add_definitions(-DENABLE_VTUNE)
    else()
        message(WARNING "VTune not found, profiling disabled")
    endif()
endif()

# Compiler warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Scheduler executable
add_executable(scheduler
    scheduler.cpp
)

target_link_libraries(scheduler
    ${ZMQ_LIBRARIES}
    pthread
)

if(TARGET nlohmann_json::nlohmann_json)
    target_link_libraries(scheduler nlohmann_json::nlohmann_json)
endif()

if(ENABLE_ONEDNN)
    target_link_libraries(scheduler dnnl)
endif()

# Install
install(TARGETS scheduler DESTINATION bin)

# Print configuration summary
message(STATUS "")
message(STATUS "Configuration Summary:")
message(STATUS "  C++ Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "  SYCL Support: ${ENABLE_SYCL}")
message(STATUS "  oneDNN Support: ${ENABLE_ONEDNN}")
message(STATUS "  VTune Support: ${ENABLE_VTUNE}")
message(STATUS "")
