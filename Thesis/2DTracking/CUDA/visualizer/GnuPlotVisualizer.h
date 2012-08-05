#ifndef GNUPLOT_VISUALIZER_H_
#define GNUPLOT_VISUALIZER_H_

#include "mesh/Mesh.h"
#include "particle/Particle.h"

class GnuPlotVisualizer
{
private:
    Mesh *mesh;
public:
    GnuPlotVisualizer(Mesh *&m);
    virtual ~GnuPlotVisualizer();
    void setStyle(std::ostream &s);
    void drawPoints(std::ostream &s);
    void drawFaces(std::ostream &s);
    void drawFaceNormals(std::ostream &s);
    void drawTrajectory(std::ostream &s, Particle particle);
    void drawParticleTracking(std::ostream &s, Particle &particle);
};

#endif
