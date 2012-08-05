#include <iostream>

#include <cuda.h>

#include "cuda/Kernel.cu"
#include "mesh/Mesh.h"
#include "mesh/MeshReader.h"

void checkCUDAError(const char *msg);
void copyToGPUAndInvoke(Mesh *mesh, Particle particle);

int main(int argc, char** argv)
{
    Mesh* mesh = readMesh("mesh");
    vector<Particle> particles = readParticles("particles", mesh);
    copyToGPUAndInvoke(mesh, particles[0]);
}

void copyToGPUAndInvoke(Mesh *mesh, Particle particle)
{   
    #include "KernelSetup.cu"

    // -------------------------------------------------------------------------
    // Invoke kernel.

    // Array just used for debugging purposes
    int nelem = 10;
    int size = sizeof(float)*nelem;
    float validation_data[nelem];
    float* validation_data_d;
    
    cudaMalloc((void**)&validation_data_d, size);
    cudaMemcpy(validation_data_d, &validation_data, size, cudaMemcpyHostToDevice);

    ParticleKernel<<<1,1>>>(validation_data_d, nelem);

    cudaMemcpy(&validation_data, validation_data_d, size, cudaMemcpyDeviceToHost);

    cout << "Printing validation data below.\n";
    for(int i=0; i<nelem; i++) {
        cout << validation_data[i] << " ";
    }
    cout << endl;

    checkCUDAError("Kernel invocation.");
    
    // TODO free memory
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
