cmake_minimum_required(VERSION 2.6)
project (ParticleTracking)

find_package(CUDA)

# for the use with nvcc-gdb
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -g)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -G)
# Use CUDA emulator?
set(CUDA_BUILD_EMULATION OFF)

# Shows nr of registers used and stuff
set(CUDA_NVCC_FLAGS --ptxas-options=-v)

if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    MESSAGE("Found Linux")
    # gtest 
    include_directories("/usr/local/include")
    link_directories("/usr/local/lib")
    # Workaround for gcc-4.4
    set(CUDA_NVCC_FLAGS --compiler-options -fpermissive)
endif()

if(APPLE)
    set(CUDA_64_BIT_DEVICE_CODE OFF)	# Does not work on a Mac currently
    set(CMAKE_C_FLAGS -m32)
    set(CMAKE_CXX_FLAGS -m32)
endif()

#include_directories(CUDA_INCLUDES)
include_directories("${PROJECT_BINARY_DIR}")
include_directories("${PROJECT_SOURCE_DIR}")


add_executable(SimpleTests
               SimpleTests.cpp
               mesh/MeshReader.cpp
               mesh/Mesh.cpp
               particle/Particle.cpp
               os/unix/fs.cpp
)

add_executable(GnuPlotVisualizer
               visualizer/GnuPlotVisualizer.cpp
               mesh/MeshReader.cpp
               mesh/Mesh.cpp
               os/unix/fs.cpp
               particle/Particle.cpp
)

cuda_add_executable(UnitTests
               unittests/StartUnittests.cpp
               unittests/Mesh_unittests.cpp
               unittests/ParticleTracker_unittests.cpp
               mesh/MeshReader.cpp
               mesh/Mesh.cpp
               particle/Particle.cpp
               os/unix/fs.cpp
)

cuda_add_executable(PerfTest
                    cuda/PerfTest.cu
                    mesh/MeshReader.cpp
                    mesh/Mesh.cpp
                    particle/Particle.cpp
                    os/unix/fs.cpp
)

target_link_libraries(UnitTests gtest)

if(APPLE)
    if(CUDA_VERSION VERSION_GREATER "2.3")
      find_library_local_first(CUDA_CUDARTEMU_LIBRARY cudartemu "\"cudartemu\" library")
      if(CUDA_BUILD_EMULATION)
        set(CUDA_LIBRARIES ${CUDA_CUDARTEMU_LIBRARY})
        if(APPLE)
          # We need to add the path to cudart to the linker using rpath, since the
          # library name for the cuda libraries is prepended with @rpath.
          get_filename_component(_cuda_path_to_cudart "${CUDA_CUDARTEMU_LIBRARY}" PATH)
          if(_cuda_path_to_cudart)
            list(APPEND CUDA_LIBRARIES -Wl,-rpath "-Wl,${_cuda_path_to_cudart}")
          endif()
        endif()

        # Add cuda library to the link line only if it is found.
        if (CUDA_CUDA_LIBRARY)
          set(CUDA_LIBRARIES ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY})
        endif(CUDA_CUDA_LIBRARY)

        mark_as_advanced(
          CUDA_CUDARTEMU_LIBRARY
          )
      endif()
    endif()
endif()
