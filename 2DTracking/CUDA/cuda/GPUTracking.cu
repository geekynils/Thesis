// -----------------------------------------------------------------------------
// Macros

// Yes the brackets around v1x are necessary.
// Otherwise the precedence of the multiplication leads to wrong results if an
// addition or subtraction of two values is put in for example v1x.

#define dotP(v1x, v1y, v2x, v2y) ((v1x)*(v2x) + (v1y)*(v2y))


// -----------------------------------------------------------------------------
// Helper functions


// -----------------------------------------------------------------------------
// Track to face

/**
 *
 * @return cell hit and lambda
 */
__device__ float2 trackToFaceBasic(
    int cell, int ignoreFace,
    float ax, float ay,
    float bx, float by,
    float *mesh, int mesh_n,
    float *validation_data
    )
{
    float Cfx, Cfy;     // face centre
    float Sx, Sy;       // face normal
    
    // Points to the memory where the data of the current cell starts.
    float *cell_data = &(mesh[cell*32]);
    
    float lambda;
    float smallest_lambda = 1;

    int face_hit = -1;
    
    // Four faces per cell
    // TODO check if the face label is not set to -1
    for(int i=0; i<4; i++)
    {
        int face = cell_data[1 + 5*i];  // Face label is at 0, face label at 1.
        Cfx = cell_data[2 + 5*i]; Cfy = cell_data[3 + 5*i];
        Sx = cell_data[4 + 5*i]; Sy = cell_data[5 + 5*i];
        
        // (Cf - a) * S
        // ------------
        // (b - a) * S
        
        lambda = dotP(Cfx - ax, Cfy - ay, Sx, Sy)
               / dotP(bx - ax, by - ay, Sx, Sy);
               
        if (lambda <= 1 && lambda >= 0)
        {
            if (lambda < smallest_lambda)
            {
            	face_hit = face;
                smallest_lambda = lambda;
            }
        }
    }
    
    return make_float2(face_hit, smallest_lambda);
}

/**
 * @param cell                  Cell in which the particle resides at the 
 *                              beginning.
 * @param cell_fraction         Stores the cell ids and the fraction of the  
 *                              concerning timesteps that the particle spent
 *                              there.
 * @param cell_fraction_index   Index over cell_fraction. Stores the start of
 *                              each time step.
 */
 /*
__device__ int trackParticle(int cell,
    float* cell_fraction, float* cell_fraction_index,
    float *validation_data, int nvalid)
{
    int face = -1;
    float lambda;
    float fractionProcessed;
    float fractionSpentInCell;
    float ax, ay, bx, by;
    int j=0;
    
    for(int i=0; i<nstepsTrajectory_d; i++)
    {
        ax = trajectoryx_d[i];
        ay = trajectoryy_d[i];
        bx = trajectoryx_d[i+1];
        by = trajectoryy_d[i+1];
        
        fractionProcessed = 0;
        
        int k=0;
        while(1)
        {
            face = trackToFaceBasic(cell, face, ax, ay, bx, by, &lambda);
            
            // Did not hit any face.
            if (face == -1)
            {
                // Just logging..
                int idx = i*k*8;
                if (!(idx + 8 >= nvalid)) {
                    validation_data[idx] = cell;
                    validation_data[idx+1] = face;
                    validation_data[idx+2] = ax;
                    validation_data[idx+3] = ay;
                    validation_data[idx+4] = bx;
                    validation_data[idx+5] = by;
                    validation_data[idx+6] = lambda;
                    validation_data[idx+7] = 666;
                }
                
                // Attach?!
                fractionSpentInCell = 1 - fractionProcessed;
                validation_data
            }
        }
    }
}
*/
