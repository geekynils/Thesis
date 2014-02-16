#include <string>

#include "mesh/Mesh.h"

using namespace std;

#ifndef MATLAB_VISUALIZER_H_
#define MATLAB_VISUALIZER_H_

class MatlabVisualizer
{
private:
	Mesh *mesh;
public:
	MatlabVisualizer(Mesh *&m);
	virtual ~MatlabVisualizer();
	string drawFaces();
	string drawParticle();
	string drawParticle2();
	string drawFaceNormals();
};
#endif
