// Fetch the data and copy it into global and constant memory.
// Note this file depends on the inclusion of GPUTracking.cu

// And also on this function.
void checkCUDAError(const char *msg);

// -----------------------------------------------------------------------------
// Fetch data into arrays.

// Points
float *pointsx, *pointsy;
int npoints = mesh->getPointsx(pointsx);
mesh->getPointsy(pointsy);

// Faces
int *facesStart, *facesEnd;
int nfaces = mesh->getFacesStart(facesStart);
mesh->getFacesEnd(facesEnd);

// Face centres
float *faceCentresx, *faceCentresy;
mesh->getFaceCentresx(faceCentresx);
mesh->getFaceCentresy(faceCentresy);

float *face_normalsx, *face_normalsy;
mesh->getFaceNormalsx(face_normalsx);
mesh->getFaceNormalsy(face_normalsy);

// Centroids
float *centroidsx, *centroidsy;
mesh->getCentroidsx(centroidsx);
mesh->getCentroidsy(centroidsy);

// Owners
// We have for every face an entry in the owners and neighbours array
// therefore its of nfaces size.
int *owners;
mesh->getOwners(owners);

// Neighbours
int *neighbours;
int nneighbours = mesh->getNeighbours(neighbours);

// Trajectory
float *trajectoryx, *trajectoryy;
int nstepsTrajectory = particle.getTrajectoryx(trajectoryx);
particle.getTrajectoryy(trajectoryy);

// Data struct for direct face access given a cell id

int *owned_faces_index;
int *owned_faces;
int *neighbour_faces_index;
int *neighbour_faces;

int n_owned_faces;
int n_neighbour_faces;

int ncells = mesh->getFacesIndexedByCells
(
    owned_faces_index,
    owned_faces,
    neighbour_faces_index,
    neighbour_faces,
    n_owned_faces,
    n_neighbour_faces
);


// -----------------------------------------------------------------------------
// Initialize global memory

// Put all the different data needed for upload in this struct attach
// them to an array.

struct toGlobalMemory
{
    void *array;
    size_t size;				// size of the array
    char *symbol;
    int sizeof_elem;            // sizeof(int) or sizeof(float)
};
typedef struct toGlobalMemory toGlobMem;

int nglobals = 18;
toGlobMem toGlobMemArray[nglobals];

toGlobMem pointsx_s =
    { pointsx, sizeof(float)*npoints, "pointsx_d", sizeof(float) };
toGlobMem pointsy_s =
    { pointsy, sizeof(float)*npoints, "pointsy_d", sizeof(float) };

toGlobMem faces_start_s =
    { facesStart, sizeof(int)*nfaces, "faces_start_d", sizeof(int) };
toGlobMem faces_end_s =
    { facesEnd, sizeof(int)*nfaces, "faces_end_d", sizeof(int) };

toGlobMem face_centresx_s =
    { faceCentresx, sizeof(float)*nfaces, "face_centresx_d", sizeof(float) };
toGlobMem face_centresy_s =
    { faceCentresy, sizeof(float)*nfaces, "face_centresy_d", sizeof(float) };
    
toGlobMem face_normalsx_s =
    { face_normalsx, sizeof(float)*nfaces, "face_normalsx_d", sizeof(float) };
toGlobMem face_normalsy_s =
    { face_normalsy, sizeof(float)*nfaces, "face_normalsy_d", sizeof(float) };

toGlobMem centroidsx_s =
    { centroidsx, sizeof(float)*nfaces, "centroidsx_d", sizeof(float) };
toGlobMem centroidsy_s =
    { centroidsy, sizeof(float)*nfaces, "centroidsy_d", sizeof(float) };

toGlobMem owners_s =
    { owners, sizeof(int)*nfaces, "owners_d", sizeof(int) };
toGlobMem neighbours_s =
    { neighbours, sizeof(int)*nneighbours, "neighbours_d", sizeof(int) };

toGlobMem owned_faces_s =
    { owned_faces, sizeof(int)*n_owned_faces, "owned_faces_d", sizeof(int) };
toGlobMem owned_faces_index_s =
    { owned_faces_index, sizeof(int)*ncells, "owned_faces_index_d", sizeof(int) };

toGlobMem neighbour_faces_s =
    { neighbour_faces, sizeof(int)*n_neighbour_faces, "neighbour_faces_d", sizeof(int) };
toGlobMem neighbour_faces_index_s =
    { neighbour_faces_index, sizeof(int)*ncells, "neighbour_faces_index_d", sizeof(int) };

toGlobMem trajectoryx_s =
    { trajectoryx, sizeof(float)*nstepsTrajectory, "trajectoryx_d", sizeof(float) };
toGlobMem trajectoryy_s =
    { trajectoryy, sizeof(float)*nstepsTrajectory, "trajectoryy_d", sizeof(float) };

toGlobMemArray[0] = pointsx_s;
toGlobMemArray[1] = pointsy_s;

toGlobMemArray[2] = faces_start_s;
toGlobMemArray[3] = faces_end_s;

toGlobMemArray[4] = face_centresx_s;
toGlobMemArray[5] = face_centresy_s;

toGlobMemArray[6] = face_normalsx_s;
toGlobMemArray[7] = face_normalsy_s;

toGlobMemArray[8] = centroidsx_s;
toGlobMemArray[9] = centroidsy_s;

toGlobMemArray[10] = owners_s;
toGlobMemArray[11] = neighbours_s;

toGlobMemArray[12] = owned_faces_s;
toGlobMemArray[13] = owned_faces_index_s;

toGlobMemArray[14] = neighbour_faces_s;
toGlobMemArray[15] = neighbour_faces_index_s;

toGlobMemArray[16] = trajectoryx_s;
toGlobMemArray[17] = trajectoryy_s;

// Loop over the array and call the concerning CUDA functions

int *scratch; // helper var
for(int i=0; i<nglobals; i++)
{
    cudaMalloc((void**)&scratch, toGlobMemArray[i].size);
    cudaMemcpy(scratch, toGlobMemArray[i].array, toGlobMemArray[i].size,
        cudaMemcpyHostToDevice);

    // Note that all pointers are of the same size!
    cudaMemcpyToSymbol(toGlobMemArray[i].symbol, &scratch, sizeof(void *), 0,
        cudaMemcpyHostToDevice);

    // cout << "Uploading: " << toGlobMemArray[i].symbol << endl;
}

checkCUDAError("Global memory allocation and uploading.");


// -----------------------------------------------------------------------------
// Initialize constant memory.

cudaMemcpyToSymbol("npoints_d", &npoints, sizeof(int), 0,
    cudaMemcpyHostToDevice);

cudaMemcpyToSymbol("nfaces_d", &nfaces, sizeof(int), 0,
    cudaMemcpyHostToDevice);

cudaMemcpyToSymbol("nstepsTrajectory_d", &nstepsTrajectory, sizeof(int), 0,
    cudaMemcpyHostToDevice);

cudaMemcpyToSymbol("n_owned_faces_d", &n_owned_faces, sizeof(int), 0,
    cudaMemcpyHostToDevice);

cudaMemcpyToSymbol("n_neighbour_faces_d", &n_neighbour_faces, sizeof(int), 0,
    cudaMemcpyHostToDevice);

cudaMemcpyToSymbol("ncells", &ncells, sizeof(int), 0, cudaMemcpyHostToDevice);

checkCUDAError("Constant memory allocation and uploading.");
