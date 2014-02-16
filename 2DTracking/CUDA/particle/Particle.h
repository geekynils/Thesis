#include <vector>

#include <gtest/gtest_prod.h>

#include "mesh/Mesh.h"

#ifndef PARTICLETRACKER_H_
#define PARTICLETRACKER_H_

class Particle
{
private:
    
    Mesh* mesh;
    
    // Constant for comparing float values
    float epsilon;

	// Track of the particle
	vector<pair<float, float> > track;
    
    // We need a label to distinguish particles
    int label;
    
    // For each time step it contains the cell labels in which the particle
    // resides and the fraction of the timestep which it spends in the cell.
    vector<pair<int, float> > cellFraction;
    
    // Maps the pairs defined above to time steps. The indices match timesteps
    // and the values the beginning index of the concerning entries in 
    // cellFraction.
    vector<int> stepCells;
    
    float dotProd(float v1x, float v1y, float v2x, float v2y);
    
#ifdef DEBUG
    void logTracking(float ax, float ay, float bx, float by, 
                    int face, float fraction, int cell);
#endif
    
    /**
     *  Fetch all the faces surrounding a cell. If ignore is set to -1,
     *  no face will be ignored, otherwise the face with the label to which
     *  ignore is set will be ignored.
     */
    vector<int> findFaces(int cell, int ignore)
    {
        vector<int> owners     (mesh->getOwners());
        vector<int> neighbours (mesh->getNeighbours());
        vector<int> faces;
        
        // Fetch all neighbour and owner faces.
        for (int i=0; i<(int)owners.size(); i++) {
            if (owners[i] == cell) {
                faces.push_back(i);
            }
        }
        
        for (int i=0; i<(int)neighbours.size(); i++) {
            if (neighbours[i] == cell) {
                faces.push_back(i);
            }
        }
        
        // Remove the face which we want to ignore.
        if (ignore != -1) {
            for (int i=0; i<(int)faces.size(); i++) {
                if(faces[i] == ignore)
                    faces.erase(faces.begin()+i);
            }
        }
        
        return faces;
    }

public:
	
    // Constructor and Deconstructor
    Particle(Mesh *&mesh_, int label_, vector<pair<float, float>  > track_);

	virtual ~Particle();

	// TODO hack should probably be private
    int trackToFaceBasic(
            int cell, int ignoreFace,
            float ax, float ay,
            float bx, float by,
            float* smallestLambda
    );

    /**
     * Calculates the cellFraction and cellSteps from the given trajectory.
     * In English: Figures out for every timestep in which cells the particle
     * was and how much time it spent there.
     */
    void trackParticle();

    int getTrajectoryx(float *&trajectoryx);
    int getTrajectoryy(float *&trajectoryy);
    
    inline vector<pair<int, float> > getCellFraction() {
        return cellFraction;
    }
    
    inline vector<int> getStepCells() {
        return stepCells;
    }
    
    inline vector<pair<float, float> > getTrajectory() {
        return track;
    }
    
    inline int getLabel() {
        return label;
    }
    
    FRIEND_TEST(testTrackToFaceBasic, trackFirstFace);
    FRIEND_TEST(testTrackToFaceBasic, trackSecondFace);
    FRIEND_TEST(testTrackToFaceBasic, trackSame);
    FRIEND_TEST(testTrackToFaceBasic, anotherTrackingTest);
    FRIEND_TEST(testTrackToFaceBasic, trackCase2);
    FRIEND_TEST(TestParticleTracker,  testDotProd);
};

#endif /* PARTICLETRACKER_H_ */
