include(FetchContent)
FetchContent_Declare(
  scran_tests
  GIT_REPOSITORY https://github.com/libscran/scran_tests
  GIT_TAG master
)
FetchContent_MakeAvailable(scran_tests)

enable_testing()

add_executable(
    libtest 
    src/simple_pca.cpp
    src/blocked_pca.cpp
    src/subset_pca.cpp
    src/utils.cpp
)

target_link_libraries(
    libtest
    scran_pca
    scran_tests
)

target_compile_options(libtest PRIVATE -Wall -Wpedantic -Wextra)

option(CODE_COVERAGE "Enable coverage testing" OFF)
if(CODE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(libtest PRIVATE -O0 -g --coverage)
    target_link_options(libtest PRIVATE --coverage)
endif()

include(GoogleTest)
gtest_discover_tests(libtest)
