#include <iostream>
#include <fstream>
#include <stdexcept>
#include <sstream>

#include <stdlib.h>

#include "mesh/Mesh.h"
#include "visualizer/MatlabVisualizer.h"
#include "particle/Particle.h"
#include "mesh/MeshReader.h"

using namespace std;

void dump()
{
	Mesh* mesh = readMesh("mesh");
 /*
    vector<pair<float,float> > points       (mesh->getPoints());
    vector<Particle>           particles    (readParticles("particles", mesh));
    vector<pair<int,int> >     faces        (mesh->getFaces());
    
    int npoints = points.size();
    int nfaces = faces.size();

    cout << "Points list" << endl;
    for(int i=0; i<npoints; i++)
        cout << mesh->getPoints()[i].first << " " << mesh->getPoints()[i].second 
             << endl;
    
    cout << endl << "Faces list" << endl;
    for(int i=0; i<nfaces; i++)
        cout << mesh->getFaces()[i].first << " " << mesh->getFaces()[i].second 
             << endl;
    
    int nparticles = particles.size();
    vector<pair<float, float> > track;
    
    cout << endl << "Particles" << endl;
    
    for(int i=0; i<nparticles; i++)
    {
        cout << "Particle " << particles[i].getLabel() << endl;
        cout << "Trajectory: ";
        
        track = particles[i].getTrajectory();
        
        for (int j=0; j<track.size(); j++) {
            cout << "(" << track[j].first << " " << track[j].second << ") ";
        }
        
        cout << endl;
    }
    
    cout << endl;

    vector<pair<float, float> > faceCentres(mesh->getFaceCentres());
    vector<pair<float, float> > faceNormals(mesh->getFaceNormals());
        
    cout << "Face centres" << endl;
    for (int i=0; i<nfaces; i++)
    {
		cout << "Face " << i << " at (" << points[faces[i].first].first << " "
             << points[faces[i].first].second << ") and ("
             << points[faces[i].second].first << " "
             << points[faces[i].second].second << "): " ;
		cout << "("<< faceCentres[i].first << " " << faceCentres[i].second << ")" 
             << endl;
	}

	cout << endl << "Face normals" << endl;
	for (int i=0; i<nfaces; i++)
	{
		cout << "Face " << i << " at (" << points[faces[i].first].first << " "
             << points[faces[i].first].second << ") and ("
             << points[faces[i].second].first << " "
             << points[faces[i].second].second << "): " ;
		cout << "("<< faceNormals[i].first << " " << faceNormals[i].second << ")" 
             << endl;
	}

	cout << endl;
	cout << "Dumping arrays for fast accessing the faces given a cell id."
		 << endl;

	int *owned_faces_index;
	int *owned_faces_a;
	int *neighbour_faces_index;
	int *neighbour_faces_a;

	int n_owned_faces;
	int n_neighbour_faces;

	int ncells = mesh->getFacesIndexedByCells
	(
		owned_faces_index,
		owned_faces_a,
		neighbour_faces_index,
		neighbour_faces_a,
		n_owned_faces,
		n_neighbour_faces
	);

	cout << "owned_faces_index" << endl;
	for(int i=0; i<ncells; i++)
	{
		cout << owned_faces_index[i] << " ";
	}

	cout << endl << "owned_faces_a" << endl;
	for(int i=0; i<n_owned_faces; i++)
	{
		cout << owned_faces_a[i] << " ";
	}

	cout << endl <<  "neighbour_faces_index" << endl;
	for(int i=0; i<ncells; i++)
	{
		cout << neighbour_faces_index[i] << " ";
	}

	cout << endl << "neighbour_faces_a" << endl;
	for(int i=0; i<n_neighbour_faces; i++)
	{
		cout << neighbour_faces_a[i] << " ";
	}
*/
	cout << endl;
    
    float* flat_mesh;
    int flat_mesh_n = mesh->getMeshFlat(&flat_mesh, false);
    for(int i=0; i<flat_mesh_n; i++)
    {
        if (i % 32 == 0)
            cout << endl;
        
        cout << flat_mesh[i] << " ";
        
        if (i % 32 == 0)
            cout << "  ";
    }
    cout << endl;
    free(flat_mesh);
    delete(mesh);
    
}


int main(int argc, const char** argv)
{
    dump();
    
    return 0;
}
