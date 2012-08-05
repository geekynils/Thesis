#include <iostream>

#include <stdlib.h>

#include <gtest/gtest.h>
#include <cuda.h>

#include "Particle.h"
#include "Mesh.h"
#include "MeshReader.h"

// Includes global and const memory decleration as well as device functions.
#include "GPUTracking.cu"

// -----------------------------------------------------------------------------
// Test CUDATrackToFace, hitFace1

// Kernel wrapping trackToFaceBasic
__global__ void kernelTrackToFaceBasic(
    float *validation_data, int n,
    int cell, int ignoreCell,
    float ax, float ay,
    float bx, float by
    )
{
    // Fill the validation data with something we know
    for (int i=0; i<100; i++) {
        validation_data[i] = 666;
    }

    // Do not forget to initialize this pointer
    float f;
    float *lambda = &f;
    int ret = trackToFaceBasic(cell, ignoreCell, ax, ay, bx, by, 
                               lambda, validation_data);
    validation_data[0] = ret;
    validation_data[1] = *lambda;
}

// -----------------------------------------------------------------------------
// Test testTrackToFaceBasic, hitFace1

TEST(CUDATrackToFace, hitFace1)
{    
    Mesh *mesh = readMesh("mesh");
    vector<Particle> particles = readParticles("particles", mesh);
    // TODO ensure that particle with label1 is at index 0
    Particle particle = particles[0];
    vector<pair<float, float> > trajectory(particle.getTrajectory());
        
    // Copies mesh and further data to the GPU
    #include "GPUSetup.cu" 

    int nelem = 100;
    int size = sizeof(float) * nelem;
    float *validation_data = (float*)malloc(size);
    float *validation_data_d;

    cudaMalloc((void**)&validation_data_d, size);
    cudaMemcpy(validation_data_d, &validation_data, size, 
               cudaMemcpyHostToDevice);

    kernelTrackToFaceBasic<<<1,1>>>(validation_data_d, nelem,
                                    0, -1,
                                    2.75, 2, 3.5, 1.5);
    
    cudaMemcpy(validation_data, validation_data_d, size, 
               cudaMemcpyDeviceToHost);
    
    /*
    cout << "Dumping validation data." << endl;
    for(int i=0; i<nelem; i++) {
        cout << validation_data[i] << endl;
    }
    */
    
    void checkCUDAError(const char *msg);
    
    ASSERT_FLOAT_EQ(1, validation_data[0])
        << "Checking if the function reports the correct face hit";
    ASSERT_TRUE(validation_data[1] >= 0);
    ASSERT_TRUE(validation_data[1] <= 1);

    // TODO free the stuff from GPUSetup
    cudaFree(validation_data_d);
    free(validation_data);
}


// -----------------------------------------------------------------------------
// Test testTrackToFaceBasic, trackSecondFace

TEST(CUDATrackToFace, hitFace1again)
{
    // Find and fetch all the required data
	Mesh *mesh = readMesh("mesh");
    vector<Particle> particles = readParticles("particles", mesh);
    // TODO ensure that particle with label1 is at index 0
    Particle particle = particles[0];
    
    // Copies mesh and further data to the GPU
    #include "GPUSetup.cu"
    
    // Validation data
    int nelem = 100;
    int size = sizeof(float) * nelem;
    float *validation_data = (float*)malloc(size);
    float *validation_data_d;
     
    cudaMalloc((void**)&validation_data_d, size);
    cudaMemcpy(validation_data_d, &validation_data, size, 
               cudaMemcpyHostToDevice);
    
    kernelTrackToFaceBasic<<<1,1>>>(validation_data_d, nelem,
                                1, -1, 2.75, 2, 3.5, 1.5);

    cudaMemcpy(validation_data, validation_data_d, size, 
               cudaMemcpyDeviceToHost);
    /*
    cout << "Dumping validation data." << endl;
    for(int i=0; i<nelem; i++) {
        cout << validation_data[i] << endl;
    }*/
    
    ASSERT_EQ(1, validation_data[0]);
    ASSERT_TRUE(validation_data[1] >= 0);
    ASSERT_TRUE(validation_data[1] <= 1);
    
    cudaFree(validation_data_d);
    free(validation_data);
}


void checkCUDAError(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if( cudaSuccess != err)
	{
		fprintf(stderr, "Cuda error: %s: %s.\n", msg,
				cudaGetErrorString( err) );
		exit(EXIT_FAILURE);
	}
}
