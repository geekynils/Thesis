#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

using namespace std;

#include "mesh/Mesh.h"
#include "mesh/MeshReader.h"
#include "particle/Particle.h"

#include "os/unix/fs.h"

Mesh* readMesh(string meshDir)
{
	vector<pair<float,float> > points;
	vector<pair<int, int> > faces;
	vector<int> owners;
    vector<int> neighbours;
	vector<pair<float,float> > particleTracks;
    vector<pair<float,float> > centroids;
    
    points = readPairs<float>(meshDir + "/points");
    faces = readPairs<int>(meshDir + "/faces");
    owners = readValues<int>(meshDir + "/owner");
    neighbours = readValues<int>(meshDir + "/neighbour");
    centroids = readPairs<float>(meshDir + "/centroids");
    
	Mesh* mesh = new Mesh(points, faces, owners, neighbours, centroids);
    
	return mesh;
}


vector<Particle> readParticles(string particleDir, Mesh *mesh)
{
    vector<string> particleFiles = listFiles(particleDir);
    
    if (particleFiles.size() == 0) {
        cout << "Could not find any particle files." << endl;
        exception();
    }
    
    vector<Particle> particles;
    vector<pair<float, float> > track;
    int label;

    for (unsigned int i=0; i<particleFiles.size(); i++)
    {
        track = readPairs<float>(particleDir + "/" + particleFiles[i]);
        istringstream ss(particleFiles[i]);
        if(ss >> label)
            particles.push_back(Particle(mesh, label, track));
        else
        {
            cout << "Filename of the particle files must be numeric."
                 << endl;
            exception();
        }
    }
    
    return particles;
}
