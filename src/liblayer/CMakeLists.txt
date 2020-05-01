set(INC_DIR "../../include/mitsuba/layer")

add_library(mitsuba-layer-obj OBJECT
  fourier.cpp       ${INC_DIR}/fourier.h
  layer.cpp         ${INC_DIR}/layer.h
  microfacet.cpp    ${INC_DIR}/microfacet.h
  phase.cpp         ${INC_DIR}/phase.h
  sparse.cpp        ${INC_DIR}/sparse.h
  storage.cpp       ${INC_DIR}/storage.h
)

add_library(mitsuba-layer SHARED $<TARGET_OBJECTS:mitsuba-layer-obj>)
set_property(TARGET mitsuba-layer-obj PROPERTY POSITION_INDEPENDENT_CODE ON)
set_target_properties(mitsuba-layer-obj mitsuba-layer PROPERTIES FOLDER mitsuba-layer)
target_compile_definitions(mitsuba-layer-obj PRIVATE -DMTS_BUILD_MODULE=MTS_MODULE_LAYER)

# Link to Intel's Thread Building Blocks
target_link_libraries(mitsuba-layer PRIVATE tbb)

# Link to Mitsuba core library
target_link_libraries(mitsuba-layer PUBLIC mitsuba-core)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/src/liblayer/cmake" "${CMAKE_SOURCE_DIR}/ext/eigen/cmake")

# Optionally link to FFTW
find_package(FFTW)

if (FFTW_FOUND)
  message(STATUS "FFTW Found:")
  message(STATUS ${FFTW_LIBRARIES})

  add_definitions("-DHAVE_FFTW")
  include_directories(${FFTW_INCLUDES})
  target_link_libraries(mitsuba-layer PUBLIC ${FFTW_LIBRARIES})
else ()
  message(STATUS "Could not find FFTW which can be used to (optionally) accelerate Fourier projections in the Layer Laboratory.")
endif()

# Optionally link to UMFPACK
find_package(Umfpack)

if (UMFPACK_LIBRARIES)
  message(STATUS "UMFPACK Found:")
  message(STATUS ${UMFPACK_LIBRARIES})

  add_definitions("-DHAVE_UMFPACK")
  include_directories(${UMFPACK_INCLUDES})
  target_link_libraries(mitsuba-layer PUBLIC ${UMFPACK_LIBRARIES})
else ()
  message(STATUS "Could not find UMFPACK which is required for Adding/Subtracting layers.")
endif()

# Copy to 'dist' directory
add_dist(mitsuba-layer)

# Python bindings
add_subdirectory(python)

# Register the test directory
add_tests(${CMAKE_CURRENT_SOURCE_DIR}/tests)
