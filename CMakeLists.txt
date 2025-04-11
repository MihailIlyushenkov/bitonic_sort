cmake_minimum_required(VERSION 3.11)
project("OPENCL_TEST")

SET(TEST_SRC
    src/jija.cpp
)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(COMPILE_FLAGS "-Wall" "-g")

add_library(TESTlib STATIC ${TEST_SRC})
target_include_directories(TESTlib PUBLIC "include")

add_executable(OPENCL_TEST 
                    src/bisort.cpp
)


target_compile_options(OPENCL_TEST PRIVATE ${COMPILE_FLAGS})
target_link_libraries(OPENCL_TEST TESTlib -lOpenCL)