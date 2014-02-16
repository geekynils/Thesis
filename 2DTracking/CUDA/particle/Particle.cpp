#include <iostream>

#include <cmath>

#include <particle/Particle.h>
#include <mesh/Mesh.h>

using namespace std;

Particle::Particle(Mesh *&mesh_, int label_, vector<pair<float, float> > track_)
: mesh(mesh_), track(track_), epsilon(1.0e-6f), label(label_)
{}

Particle::~Particle()
{}

/**
 * Returns the face index crossed by the particle -1 if the particle stays in 
 * the current cell.
 *
 * @param cell Cell label of the cell in which the particle resides at the 
 *             beginning of the step.
 *
 * @param lambda Fraction of the timestep processed, if the particle changes the 
 *               cell, otherwise it's something else.
 *
 * @param ignoreFace While tracking down the particle ensure that the face,
 *                   hit before is ignored.
 */

int Particle::trackToFaceBasic(
    int cell, int ignoreFace,
    float ax, float ay, 
    float bx, float by,
    float *smallestLambda
)
{
    vector<pair<float, float> > faceCentres (mesh->getFaceCentres());
    vector<pair<float, float> > faceNormals (mesh->getFaceNormals());
    
    // Vector holding all faces surrounding a concerning cell.
    vector<int> faces(findFaces(cell, ignoreFace));
    int nfaces = faces.size();
    
    float Cfx, Cfy;     // face centre
    float Sx, Sy;       // face normal
    
    float lambda;
    
    // Smallest lambda found so far.
    *smallestLambda = 1;
    
    // -1 if it stays in the same cell
    int faceWithSmallestLambda = - 1;
    
    // Iteration over all faces belonging to a cell
    for (int i=0; i<nfaces; i++)
    {
        // Fetch Cf and Sf
        Cfx = faceCentres[faces[i]].first; Cfy = faceCentres[faces[i]].second;
        Sx = faceNormals[faces[i]].first; Sy = faceNormals[faces[i]].second;

        if (!mesh->isOwner(cell, i)) {
            Sx *= -1; Sy *= -1;
        }
        
        // (Cf - a) * S
        // -------------
        // (b - a)  * S
        
        lambda = dotProd(Cfx - ax, Cfy - ay, Sx, Sy) 
                 / dotProd(bx - ax, by - ay, Sx, Sy);
        
        // Debug
        // cout << "Lambda (face " << faces[i] << ") " << lambda << endl;
        
        if ((lambda <= 1 && lambda >= 0))
        {
            if(lambda < *smallestLambda)
            {
                faceWithSmallestLambda = faces[i];
                *smallestLambda = lambda;
            }
        }
    }
    
    return faceWithSmallestLambda;
    
}

float Particle::dotProd(float v1x, float v1y, float v2x, float v2y)
{
    return (v1x * v2x + v1y * v2y);
}


/**
 * @param cell Cell in which the particle resides at the beginning.
 *
 */
void Particle::trackParticle()
{
    
// --- Initialize data
    
    int cell = mesh->findCell(track[0].first, track[0].second);
    
    // vector<pair<float, float> > particleTrack   (mesh->getParticleTrack());
    vector<int>                 owners          (mesh->getOwners());
    vector<int>                 neighbours      (mesh->getNeighbours());
    
    // Number of steps = Number of positions - 1.
    // First step is from position 0 to 1, then from 1 to 0 until from n-1 to n.
    int nsteps = track.size() - 1;
    
    // face crossed by the particle
    int face = -1;
    
    float lambda;
    
    // Total fraction of the timestep which has already been processed.
    float fractionProcessed;
    
    // Fraction of timestep which the particle spent in the current cell.
    float fractionSpentInCell;
    
    // Start and end of a step.
    float ax, ay, bx, by;
    
    // Index of the cellFraction array.
    int j = 0;
    
// -- Algorithm starts below
    
#ifdef DEBUG
    cout << endl << "Start tracking particle from cell " << cell << "." 
         << endl << endl;
#endif
    
    // Loop over all timesteps.
    for (int i=0; i<nsteps; i++)
    {
        ax = track[i].first;
        ay = track[i].second;
        bx = track[i+1].first;
        by = track[i+1].second;
        
        fractionProcessed = 0;
        
        stepCells.push_back(j);
        
#ifdef DEBUG
        cout << endl << "Timestep " << i << endl << endl;
#endif
        
        // Track the particle down.
        while (true) 
        {
            face = trackToFaceBasic(cell, face, ax, ay, bx, by, &lambda);
            
            // If this is the case, the particle stays in the current cell and
            // we are done tracking it.
            if (face == -1)
            {
                fractionSpentInCell = 1 - fractionProcessed;
#ifdef DEBUG
                logTracking(ax, ay, bx, by, face, fractionSpentInCell, cell);
#endif
                cellFraction.push_back(make_pair(cell, fractionSpentInCell));
                j++;
                break;
            }
            
            // The lambda given back by trackToFace relates to only the fraction
            // of the timestep not yet processed! We need to calculate there the
            // fraction of the whole timestep it actually spent in the cell.
            if(fractionProcessed == 0) {
                fractionSpentInCell = lambda;
            } else {
                fractionSpentInCell = (1 - fractionProcessed) *lambda;
            }

            fractionProcessed += fractionSpentInCell;
        
#ifdef DEBUG
            logTracking(ax, ay, bx, by, face, fractionSpentInCell, cell);
#endif
            cellFraction.push_back(make_pair(cell, fractionSpentInCell));
            
            // Calculate the vector from the position where the particle hits
            // the face to the end of the time step.
            // a + lambda * (b - a)
            ax = ax + lambda * (bx - ax);
            ay = ay + lambda * (by - ay);
            
            // If the current cell is owner of the face.
            if (mesh->isOwner(cell, face)) 
            {
                // Then the particle will be in the concerning neighbour cell.
                cell = neighbours[face];
            } else {
                cell = owners[face];
            }

            j++;
        }
    }
}

#ifdef DEBUG
void Particle::logTracking(float ax, float ay, float bx, float by, 
                                  int face, float fraction, int cell)
{
    cout << "Tracking the particle from a (" << ax << " " << ay << ") to (" 
         << bx << " " << by << ")" << " in cell " << cell
         << " fraction of the timestep it spent in cell: " << fraction << endl;
    
    if (face != -1)
        cout << "Face " << face << " was hit." << endl;

    
    cout << endl;
}
#endif


int Particle::getTrajectoryx(float *&trajectoryx)
{
    int npoints = track.size(); 
    trajectoryx = new float[npoints];
    for(int i=0; i<npoints; i++) {
        trajectoryx[i] = track[i].first;
    }
    return npoints;
}

int Particle::getTrajectoryy(float *&trajectoryy)
{
    int npoints = track.size(); 
    trajectoryy = new float[npoints];
    for(int i=0; i<npoints; i++) {
        trajectoryy[i] = track[i].second;
    }
    return npoints;
}
