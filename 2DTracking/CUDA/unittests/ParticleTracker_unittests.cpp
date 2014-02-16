#include <vector>
#include <utility>

#include <gtest/gtest.h>

#include "particle/Particle.h"
#include "mesh/Mesh.h"
#include "mesh/MeshReader.h"


TEST(testTrackToFaceBasic, trackFirstFace)
{
	Mesh *mesh = readMesh("mesh");
    vector<Particle> particles = readParticles("particles", mesh);

    // Find Particle with label 1
    Particle *particle;
    for(int i=0; i<particles.size(); i++) {
        if(particles[i].getLabel() == 1) {
            particle = &particles[i];
            break;
        }
    }

    vector<pair<float, float> > trajectory(particle->getTrajectory());
    float lambda;
    
    ASSERT_EQ(1, particle->trackToFaceBasic(0, -1,
        trajectory[0].first, trajectory[0].second, 
        trajectory[1].first, trajectory[1].second,
        &lambda)
    );
    ASSERT_TRUE(lambda >= 0);
    ASSERT_TRUE(lambda <= 1);
}


TEST(testTrackToFaceBasic, trackSecondFace)
{
	Mesh *mesh = readMesh("mesh");
    vector<Particle> particles = readParticles("particles", mesh);

    Particle *particle;
    for(int i=0; i<particles.size(); i++) {
        if(particles[i].getLabel() == 1) {
            particle = &particles[i];
            break;
        }
    }

    vector<pair<float, float> > trajectory(particle->getTrajectory());
    float lambda;
    
    ASSERT_EQ(1, particle->trackToFaceBasic(1, -1,
        trajectory[1].first, trajectory[1].second, 
        trajectory[2].first, trajectory[2].second,
        &lambda)
    );
    ASSERT_TRUE(lambda >= 0);
    ASSERT_TRUE(lambda <= 1);
}

// Particle stays in the same cell
TEST(testTrackToFaceBasic, trackSame)
{
	Mesh *mesh = readMesh("mesh");
    vector<Particle> particles = readParticles("particles", mesh);

    Particle *particle;
    for(int i=0; i<particles.size(); i++) {
        if(particles[i].getLabel() == 1) {
            particle = &particles[i];
            break;
        }
    }

    vector<pair<float, float> > trajectory(particle->getTrajectory());
    float lambda;
    
    ASSERT_EQ(-1, particle->trackToFaceBasic(1, -1, 1.5, 3, 2.5, 1.5, &lambda)
    );
}

TEST(testTrackToFaceBasic, anotherTrackingTest)
{
	Mesh *mesh = readMesh("mesh");
    vector<Particle> particles = readParticles("particles", mesh);

    Particle *particle;
    for(int i=0; i<particles.size(); i++) {
        if(particles[i].getLabel() == 1) {
            particle = &particles[i];
            break;
        }
    }

    vector<pair<float, float> > trajectory(particle->getTrajectory());
    float lambda;
    
    ASSERT_EQ(2, particle->trackToFaceBasic(0, -1, 3.4, 1.79, 3, 3, &lambda));
    ASSERT_TRUE(lambda >= 0);
    ASSERT_TRUE(lambda <= 1);
}

/*
 * Tracking the particle from a (3.23239 1.11972) to (3.4 1) in cell 1 fraction of 
 * the timestep it spent in cell: 7.0123e-07
 * Face 1 was hit.
 */

TEST(testTrackToFaceBasic, trackCase2)
{
	Mesh *mesh = readMesh("mesh");
    vector<Particle> particles = readParticles("particles", mesh);
    
    Particle *particle;
    for(int i=0; i<particles.size(); i++) {
        if(particles[i].getLabel() == 2) {
            particle = &particles[i];
            break;
        }
    }

    vector<pair<float, float> > trajectory(particle->getTrajectory());
    float lambda;

    int faceHit = particle->trackToFaceBasic
        (1, 1, 3.23239, 1.11972, 3.4, 1, &lambda); 
    
    ASSERT_EQ(-1, faceHit);
    ASSERT_TRUE(lambda >= 0);
    ASSERT_TRUE(lambda <= 1);
}

TEST(TestParticleTracker, testDotProd)
{
	Mesh *mesh = readMesh("mesh");
    vector<Particle> particles = readParticles("particles", mesh);
    ASSERT_FLOAT_EQ(17, particles[0].dotProd(3, 2, 1, 7));
}


TEST(TestParticleTracker, testTrackParticle)
{
   	Mesh *mesh = readMesh("mesh");
    vector<Particle> particles = readParticles("particles", mesh);

    Particle *particle;
    for(int i=0; i<particles.size(); i++) {
        if(particles[i].getLabel() == 1) {
            particle = &particles[i];
            break;
        }
    }

    vector<pair<float, float> > trajectory(particle->getTrajectory());
    
    particle->trackParticle();
    
    vector<pair<int, float> > cellFraction(particle->getCellFraction());
    vector<int> stepCells(particle->getStepCells());
    
    ASSERT_TRUE(stepCells.size() == 2);
    ASSERT_TRUE(cellFraction.size() == 5);
    ASSERT_TRUE(cellFraction[cellFraction.size()-1].first == 2);
}


TEST(TestParticleTracker, testTrackParticle2)
{
   	Mesh *mesh = readMesh("mesh");
    vector<Particle> particles = readParticles("particles", mesh);

    Particle *particle;
    for(int i=0; i<particles.size(); i++) {
        if(particles[i].getLabel() == 2) {
            particle = &particles[i];
            break;
        }
    }

    vector<pair<float, float> > trajectory(particle->getTrajectory());
    
    particle->trackParticle();
    
    vector<pair<int, float> > cellFraction(particle->getCellFraction());
    vector<int> stepCells(particle->getStepCells());
    
    ASSERT_TRUE(stepCells.size() == 3);
    ASSERT_EQ(6, cellFraction.size());
    ASSERT_EQ(2, cellFraction[cellFraction.size()-1].first);
}
