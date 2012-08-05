#include <iostream>
#include <sstream>
#include <stdexcept>

#include "mesh/MatlabVisualizer.h"

// Visualize a line with the following matlab command
// line([.3 .7],[.4 .9],'Marker','.','LineStyle','-', 'Color', [0 0 0]);

using namespace std;

MatlabVisualizer::MatlabVisualizer(Mesh *&m): mesh(m)
{}

MatlabVisualizer::~MatlabVisualizer()
{}

// Note strings in C++ are mutable
string MatlabVisualizer::drawFaces()
{
    ostringstream stm;
    const string start = "line(";
    const string end = ", 'LineStyle', '-', 'Color', [0 0 0]);";
    float point1x, point1y,
          point2x, point2y;
    
	vector<pair<float,float> > points = mesh->getPoints();
    vector<pair<int,int> > faces = mesh->getFaces();    
    int nfaces = faces.size();
    
    for(int i=0; i<nfaces; i++)
    {
        point1x = points[faces[i].first].first;
        point1y = points[faces[i].first].second;
        point2x = points[faces[i].second].first;
        point2y = points[faces[i].second].second;
        
        stm << start << "[ ";
        stm << point1x << " " << point2x;
        stm << "], [";
        stm << point1y << " " << point2y;
        stm << "]" << end << endl;
    }

    return stm.str();
}


string MatlabVisualizer::drawParticle()
{
	// From the matlab help
	// quiver(x,y,u,v)
	// A quiver plot displays velocity vectors as arrows 
    // with components (u,v) at the points (x,y).

	ostringstream stm;
	const string start = "quiver([";
	const string end = "],0);";

    vector<pair<float, float> > particleTrack(mesh->getParticleTrack());
    
	stm << start;

    int n = particleTrack.size()-1;
    
	for(int i=0; i<n; i++) {
		stm << particleTrack[i].first << " ";
	}
	stm << "], [";
	for(int i=0; i<n; i++) {
		stm << particleTrack[i].second << " ";
	}
	stm << "], [";
	for(int i=0; i<n; i++) {
		stm << (particleTrack[i+1].first - particleTrack[i].first) << " ";
	}
	stm << "], [";
	for(int i=0; i<n; i++) {
		stm << (particleTrack[i+1].second - particleTrack[i].second) << " ";
	}
	stm << end << endl;

	return stm.str();
}
/*
string MatlabVisualizer::drawParticle2()
{
	// line([ 1 3], [1 0.5], 'Marker', '.', 'LineStyle', '-', 'Color', [0 0 0]);

	ostringstream stm;
	const string start = "line([";
	const string end = "], 'Marker', '.', 'LineStyle', '-', 'Color', [1 0 0]);";

	for(int i=0; i<mesh->nparticleSteps-1; i++)
	{
		stm << start;
		stm << mesh->particlex[i] << " ";
		stm << mesh->particlex[i+1] << "], [";
		stm << mesh->particley[i] << " ";
		stm << mesh->particley[i+1] << end << endl;
	}

	return stm.str();
}
*/

string MatlabVisualizer::drawFaceNormals()
{
    /*
    From the matlab help
	quiver(x,y,u,v)
	A quiver plot displays velocity vectors 
    as arrows with components (u,v) at the points (x,y).
    */

	ostringstream stm;
	const string start = "quiver([";
	const string end = "], 0.1, 'Color', [0 0 0]);";

	stm << start;
    
    vector<pair<int, int> > faces(mesh->getFaces());;
    vector<pair<float, float> > faceCentres(mesh->getFaceCentres());
    vector<pair<float, float> > faceNormals(mesh->getFaceNormals());
    
    int nfaces = faces.size();

	for(int i=0; i<nfaces; i++) {
		stm << faceCentres[i].first << " ";
	}

	stm << "], [";

	for(int i=0; i<nfaces; i++) {
		stm << faceCentres[i].second << " ";

	}
	stm << "], [";

	for(int i=0; i<nfaces; i++) {
		stm << faceNormals[i].first << " ";
	}

	stm << "], [";

	for(int i=0; i<nfaces; i++) {
		stm << faceNormals[i].second << " ";
	}

	stm << end;
	return stm.str();
}
