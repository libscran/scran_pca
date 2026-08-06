cmake_minimum_required(VERSION 3.14)

project(
    scran_pca
    VERSION 0.5.0
    DESCRIPTION "Principal components analysis for single-cell data"
    LANGUAGES CXX
)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Library
add_library(scran_pca INTERFACE)
add_library(libscran::scran_pca ALIAS scran_pca)

target_include_directories(scran_pca INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/scran_pca>)
target_compile_features(scran_pca INTERFACE cxx_std_17)

# Dependencies
option(SCRAN_PCA_FETCH_EXTERN "Automatically fetch scran_pca's external dependencies." ON)
if(SCRAN_PCA_FETCH_EXTERN)
    add_subdirectory(extern)
else()
    find_package(tatami_tatami 4.1.0 CONFIG REQUIRED)
    find_package(tatami_tatami_stats 3.1.0 CONFIG REQUIRED)
    find_package(tatami_tatami_mult 0.2.0 CONFIG REQUIRED)
    find_package(libscran_scran_blocks 0.1.0 CONFIG REQUIRED)
    find_package(ltla_irlba 3.1.0 CONFIG REQUIRED)
    find_package(libscran_irlba_tatami 0.1.0 CONFIG REQUIRED)
    find_package(ltla_sanisizer 0.2.0 CONFIG REQUIRED)
    find_package(Eigen3 5.0.0 CONFIG REQUIRED)
endif()

target_link_libraries(
    scran_pca INTERFACE 
    tatami::tatami
    tatami::tatami_stats
    tatami::tatami_mult
    libscran::scran_blocks
    ltla::irlba
    libscran::irlba_tatami
    Eigen3::Eigen
)

# Tests
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    option(SCRAN_PCA_TESTS "Build scran_pca's test suite." ON)
else()
    option(SCRAN_PCA_TESTS "Build scran_pca's test suite." OFF)
endif()

if(SCRAN_PCA_TESTS)
    include(CTest)
    if(BUILD_TESTING)
        add_subdirectory(tests)
    endif() 
endif()

# Install
install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/scran_pca)

install(TARGETS scran_pca
    EXPORT scran_pcaTargets)

install(EXPORT scran_pcaTargets
    FILE libscran_scran_pcaTargets.cmake
    NAMESPACE libscran::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/libscran_scran_pca)

configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Config.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/libscran_scran_pcaConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/libscran_scran_pca)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/libscran_scran_pcaConfigVersion.cmake"
    COMPATIBILITY SameMajorVersion)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/libscran_scran_pcaConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/libscran_scran_pcaConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/libscran_scran_pca)
