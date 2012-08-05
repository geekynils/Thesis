#include <iostream>
#include <vector>
#include <map>

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <ctime>

#include <cuda.h>

#include "particle/Particle.h"
#include "mesh/Mesh.h"
#include "mesh/MeshReader.h"

#include "cuda/GPUTracking.cu"

using namespace std;

void checkCUDAError(const char *msg);

// -----------------------------------------------------------------------------
// Kernel

__global__ void kernel(float *validation_data, int n, 
                       int* start_cells, float *x, int x_n,
                       float* mesh, int mesh_n
                       )
{
    float2 face_lambda;
    
    // 4 Particles per thread
    // Index referring to the current particle

    int idx;
    for(int i=0; i<4; i++)
    {
        // 524 288 / 4 = 131072
        idx = threadIdx.x + (blockIdx.x * blockDim.x) + 131072*i;
        
        face_lambda = trackToFaceBasic(start_cells[idx], -1,
									   x[idx*4], x[idx*4+1],
									   x[idx*4+2], x[idx*4+3],
									   mesh, mesh_n,
									   validation_data);
        // Per particle 7 values
        // Starting cell, timestep (4 floats), face hit, lambda
        validation_data[idx*7]     = start_cells[idx];
        validation_data[idx*7 + 1] = x[idx*4];
        validation_data[idx*7 + 2] = x[idx*4 + 1];
        validation_data[idx*7 + 3] = x[idx*4 + 2];
        validation_data[idx*7 + 4] = x[idx*4 + 3];
        validation_data[idx*7 + 5] = face_lambda.x;
        validation_data[idx*7 + 6] = face_lambda.y;
    }
}

void dump_validation_data(float* validation_data, int n)
{
	cout << endl << "Validation data" << endl;
	for(int i=0; i<n; i++) {
		if(i % 7 == 0)
			cout << endl;
		cout << validation_data[i] << " ";
	}
	cout << endl;
}

map<string, string> parse_args(int argc, char** argv)
{
	map<string, string> config;

    vector<string> cmdline(argv, argv+argc);

    for(size_t i=0; i<cmdline.size(); i++)
    {
        if(cmdline[i] == "-d" || cmdline[i] == "--device")
        {
            if(i+1 < cmdline.size())
            {
                config["deviceName"] = cmdline[i+1];
            }
        }

        if(cmdline[i] == "-a" || cmdline[i] == "--data")
        {
            if(i+1 < cmdline.size())
            {
                config["data"] = cmdline[i+1];
            }
        }

        if(cmdline[i] == "-h" || cmdline[i] == "--help")
        {
        	cout << "Usage: " << argv[0]
        	     << "--device <deviceName> --data <dataDir>" << endl;
        	exit(EXIT_SUCCESS);
        }
    }

    return config;
}


int main(int argc, char** argv)
{
    // -------------------------------------------------------------------------
    // Find and set CUDA device

    int num_devices, device;
    cudaGetDeviceCount(&num_devices);

    map<string, string> conf = parse_args(argc, argv);

    if (num_devices > 0)
    {
        if(conf.find("deviceName") != conf.end())
        {
			for (device = 0; device < num_devices; device++)
			{
				cudaDeviceProp props;
				cudaGetDeviceProperties(&props, device);
				if (strcmp(props.name, conf["deviceName"].c_str()) == 0)
				{
					printf("Found device!\n");
					printf("Device %d: \"%s\" with Compute %d.%d capability\n",
							device, props.name, props.major, props.minor);
					printf("Processor Count on GPU: %i\n\n",
							props.multiProcessorCount);
					cudaSetDevice(device);
					break;
				}
			}
        } else {
        	cout << "No device arguments provided, just using the first one."
        		 << endl;
        }
    } else {
    	fprintf(stderr, "No device found!\n");
    	exit(EXIT_FAILURE);
    }

    checkCUDAError("Could not set device!\n");

    if(conf.find("data") == conf.end())
    {
    	cout << "Please provide the datadir using --data" << endl;
    	exit(EXIT_FAILURE);
    }

    // -------------------------------------------------------------------------
    // Mesh data
    
    Mesh *mesh = readMesh(conf["data"] + "/mesh");
    float* mesh_flat;
    int mesh_flat_n = mesh->getMeshFlat(&mesh_flat, false);
    float scratch1 = 0;
    float *mesh_flat_d = &scratch1;
    
    // Note static_cast is not allowed here.
    cudaMalloc(reinterpret_cast<void**>(&mesh_flat_d),
        sizeof(float)* mesh_flat_n);
    cudaMemcpy(mesh_flat_d, mesh_flat, sizeof(float)* mesh_flat_n, 
        cudaMemcpyHostToDevice);
    
    checkCUDAError("Error after uploading mesh data\n");
    
    // -------------------------------------------------------------------------
    // Particle data

    vector<pair<float, float> > points(readPairs<float>(conf["data"] + "/points"));
    
    if(points.size() != 1048576) {
        // We have two points per particle movement
        // => 524 288 Particles
        cout << "Got not the expected number of particles." << endl;
        EXIT_FAILURE;
    }
    
    float* points_flat;
    float scratch2;
    float* points_flat_d = &scratch2;
    int points_flat_n = points.size()*2;
    points_flat = reinterpret_cast<float*>(malloc(sizeof(float)*points_flat_n));
    
    for(unsigned int i=0; i<points_flat_n/2; i++)
    {
        points_flat[i*2] = points[i].first;
        points_flat[i*2+1] = points[i].second;
    }
    
    // Four floats per step, find the cell in which it starts.
    // (that is in which the first point resides)
    int start_cells_n = points_flat_n/4;
    int *start_cells = reinterpret_cast<int*>(malloc(start_cells_n*sizeof(int)));
    int *start_cells_d;
    int cell;
    for (int i=0; i<start_cells_n; i++)
    {
        cell = mesh->findCell(points_flat[i*4], points_flat[i*4+1]);
        start_cells[i] = cell;
        // We only have three cells in this mesh.
        if(!(cell == 0 || cell == 1 || cell == 2))
        {
            cout << "Point not found: (" << points_flat[i*4] << " " 
                 << points_flat[i*4+1] << ")" << endl;
            cout << "Cell reported: " << cell << endl;
        }
        
        start_cells[i] = cell;
    }

    cudaMalloc(reinterpret_cast<void**>(&points_flat_d),
    	sizeof(float)*points_flat_n);
    cudaMalloc(reinterpret_cast<void**>(&start_cells_d),
    	sizeof(int)*start_cells_n);
    
    cudaMemcpy(points_flat_d, points_flat, sizeof(float)*points_flat_n,
        cudaMemcpyHostToDevice);
    cudaMemcpy(start_cells_d, start_cells, sizeof(int)*start_cells_n,
        cudaMemcpyHostToDevice);

    checkCUDAError("Error after uploading particle data\n");
    
    // -------------------------------------------------------------------------
    // Validation data
    
    // 7 Entries per particle
    int validation_data_len = start_cells_n * 7;
    int size = validation_data_len*sizeof(float);
    float *validation_data = reinterpret_cast<float*>(malloc(size));
    float scratch3;
    float *validation_data_d = &scratch3;

    cudaMalloc(reinterpret_cast<void**>(&validation_data_d), size);
    cudaMemcpy(validation_data_d, validation_data, size, cudaMemcpyHostToDevice);

    checkCUDAError("Error after uploading test specific data\n");

    // -------------------------------------------------------------------------
    // Kernel invocation

    cout << "Invoking the CUDA kernel.." << endl;

    // Blocks, Threads per blocks

    kernel<<<512,256>>>(validation_data_d, validation_data_len, 
                        start_cells_d, points_flat_d, points_flat_n,
                        mesh_flat_d, mesh_flat_n);

    cudaMemcpy(validation_data, validation_data_d, size, cudaMemcpyDeviceToHost);
    // dump_validation_data(validation_data, 700);

    // -------------------------------------------------------------------------
    // CPU version

    cout << "Starting validation on the CPU.." << endl;

    float *validation_data_cpu = reinterpret_cast<float*>(malloc(size));
    // TODO Create particle objects out of all the the time steps
    string particleDirectory = conf["data"] + "/particles";
    vector<Particle> particles(readParticles(particleDirectory, mesh));
    Particle particle = particles[0];

    float f;
    float* lambda = &f;
    int face_hit;

    clock_t cpu_start;
    clock_t cpu_end;

    cpu_start = clock();

    for(int i=0; i<start_cells_n; i++)
    {
    	face_hit = particle.trackToFaceBasic(start_cells[i], -1,
						points[i*2].first, points[i*2].second,
						points[i*2+1].first, points[i*2+1].second,
						lambda);

    	validation_data_cpu[i*7] 	 = start_cells[i];
    	validation_data_cpu[i*7 + 1] = points[i*2].first;
    	validation_data_cpu[i*7 + 2] = points[i*2].second;
    	validation_data_cpu[i*7 + 3] = points[i*2+1].first;
    	validation_data_cpu[i*7 + 4] = points[i*2+1].second;
    	validation_data_cpu[i*7 + 5] = face_hit;
    	validation_data_cpu[i*7 + 6] = *lambda;
    }

    cpu_end = clock();

    double cpu_execution_time = ((double)(cpu_end - cpu_start))/((double)CLOCKS_PER_SEC);

    cout << "Sequential execution on the CPU took " << cpu_execution_time
    	 << " seconds." << endl;

    // -------------------------------------------------------------------------
    // Result validation

    cout << "Validating results.." << endl;

    float epsilon = 10e-6;
    float diff;

    for(int i=0; i<validation_data_len; i++)
    {
    	diff = validation_data_cpu[i] - validation_data[i];
    	if(diff*diff > 10e-6)
    	{
    		cout << "GPU and CPU calc are not equal at pos " << i << endl;
    		EXIT_FAILURE;
    	}
    }

    cout << "Validation data looks ok :)" << endl;


    // -------------------------------------------------------------------------
    // Freeing memory

    free(mesh_flat);
    free(points_flat);
    free(start_cells);
    free(validation_data);
    free(validation_data_cpu);

    delete(mesh);

    cudaFree(mesh_flat_d);
    cudaFree(points_flat_d);
    cudaFree(start_cells_d);
    cudaFree(validation_data_d);

    return 0;
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
