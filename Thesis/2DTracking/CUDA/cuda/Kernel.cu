#include <cuda.h>
#include <stdio.h>

#ifndef PARTICLE_KERNEL_CU
#define PARTICLE_KERNEL_CU

// -----------------------------------------------------------------------------
// Macros

#define dotP(v1x, v1y, v2x, v2y) (v1x*v2x + v1y*v2y)

// -----------------------------------------------------------------------------
// Global memory
__device__ float *pointsx_d;
__device__ float *pointsy_d;

__device__ int *faces_start_d;
__device__ int *faces_end_d;

__device__ float *face_centresx_d;
__device__ float *face_centresy_d;

__device__ float *face_normalsx_d;
__device__ float *face_normalsy_d;

__device__ float *centroidsx_d;
__device__ float *centroidsy_d;

__device__ int *owners_d;
__device__ int *neighbours_d;

__device__ int *owned_faces_d;
__device__ int *owned_faces_index_d;

__device__ int *neighbour_faces_d;
__device__ int *neighbour_faces_index_d;

__device__ float *trajectoryx_d;
__device__ float *trajectoryy_d;

// -----------------------------------------------------------------------------
// Constant memory
__constant__ int npoints_d;
__constant__ int nfaces_d;
__constant__ int nstepsTrajectory_d;
__constant__ int n_owned_faces_d;
__constant__ int n_neighbour_faces_d;
__constant__ int ncells;


__device__ int trackToFaceBasic(
    int cell, int ignoreFace,
    float ax, float ay,
    float bx, float by,
    float *smallestLambda
    )
{
    float Cfx, Cfy;     // face centre
    float Sx, Sy;       // face normal
    float lambda;
    
    *smallestLambda = 1;

    int faceWithSmallestLambda = -1;
    
    // Iteration over all owner cells
    // TODO ncells
    for (int i=owned_faces_index_d[cell]; i<owned_faces_index_d[cell+1]; i++)
    {
        int face = owned_faces_d[i];
        Cfx = face_centresx_d[i]; Cfy = face_centresy_d[i];
        Sx = face_normalsx_d[i]; Sy = face_normalsy_d[i];
        
        // (Cf - a) * S
        // ------------
        // (b - a) * S
        
        lambda = dotP(Cfx - ax, Cfy - ay, Sx, Sy)
               / dotP(bx - ax, by - ay, Sx, Sy);
        
        if (lambda <= 1 && lambda >= 0) {
            if (lambda < *smallestLambda) {
                faceWithSmallestLambda = owners_d[i];
                *smallestLambda = lambda;
            }
        }
    }
    
    return faceWithSmallestLambda;
}

// TODO pass the data relevant for one particle
// * Mesh
// * Array with surrounding owner and neighbour faces
__global__ void ParticleKernel(float *validation_data, int n)
{
    validation_data[0] = pointsx_d[0];
    validation_data[1] = pointsx_d[1];
	printf("%f\n", pointsx_d[0]);
}
#endif
