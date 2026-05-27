cmake_minimum_required(VERSION 3.15)
project(privacy_cpp C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

add_library(privacy_filter_core STATIC
    src/privacy_filter.c
)

target_include_directories(privacy_filter_core PUBLIC src)

add_executable(privacy_filter
    src/main.c
)
target_link_libraries(privacy_filter PRIVATE privacy_filter_core)
set_target_properties(privacy_filter PROPERTIES OUTPUT_NAME privacy)

add_executable(privacy_filter_tests
    tests/test_privacy_filter.c
)
target_link_libraries(privacy_filter_tests PRIVATE privacy_filter_core)

add_executable(pf_bench
    tools/pf_bench.c
)
target_link_libraries(pf_bench PRIVATE privacy_filter_core)
set_target_properties(pf_bench PROPERTIES OUTPUT_NAME privacy-bench)

enable_testing()
add_test(NAME privacy_filter_tests COMMAND privacy_filter_tests --model ${CMAKE_SOURCE_DIR}/model)

option(PF_USE_AVX2 "Enable AVX2 SIMD optimizations" ON)
option(PF_USE_OPENMP "Enable OpenMP multi-threading" ON)

if (PF_USE_OPENMP)
    find_package(OpenMP)
endif()

if (MSVC)
    set(PF_OPT_FLAGS /O2 /W4 /fp:fast)
    if (PF_USE_AVX2)
        list(APPEND PF_OPT_FLAGS /arch:AVX2)
    endif()
else()
    set(PF_OPT_FLAGS -O3 -Wall -Wextra -ffast-math)
    if (PF_USE_AVX2)
        list(APPEND PF_OPT_FLAGS -mavx2 -mfma)
    endif()
    target_link_libraries(privacy_filter_core PUBLIC m)
endif()

if (PF_USE_AVX2)
    target_compile_definitions(privacy_filter_core PRIVATE PF_USE_AVX2=1)
    target_compile_definitions(privacy_filter PRIVATE PF_USE_AVX2=1)
    target_compile_definitions(privacy_filter_tests PRIVATE PF_USE_AVX2=1)
    target_compile_definitions(pf_bench PRIVATE PF_USE_AVX2=1)
endif()

target_compile_options(privacy_filter_core PRIVATE ${PF_OPT_FLAGS})
target_compile_options(privacy_filter PRIVATE ${PF_OPT_FLAGS})
target_compile_options(privacy_filter_tests PRIVATE ${PF_OPT_FLAGS})
target_compile_options(pf_bench PRIVATE ${PF_OPT_FLAGS})

if (PF_USE_OPENMP AND OpenMP_C_FOUND)
    target_link_libraries(privacy_filter_core PUBLIC OpenMP::OpenMP_C)
endif()

install(TARGETS privacy_filter pf_bench
    RUNTIME DESTINATION bin
)
install(FILES src/privacy_filter.h
    DESTINATION include
)

