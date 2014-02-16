#include <sstream>
#include <vector>

#include "visualizer/GnuPlotVisualizer.h"
#include "mesh/Mesh.h"
#include "mesh/MeshReader.h"
#include "particle/Particle.h"

GnuPlotVisualizer::GnuPlotVisualizer(Mesh *&m)
: mesh(m)
{}

GnuPlotVisualizer::~GnuPlotVisualizer()
{}

void GnuPlotVisualizer::setStyle(std::ostream &s)
{
    s << "unset arrow\n";
    s << "unset mouse\n";
    s << "set title 'Mesh' font 'Arial,12'\n";
    s << "set style line 1 pointtype 7 linecolor rgb 'gray'\n";
    s << "set style line 2 pointtype 7 linecolor rgb 'black'\n";
    s << "set style line 3 pointtype 7 linecolor rgb 'red'\n";
    s << "set pointsize 2\n";
}

void GnuPlotVisualizer::drawPoints(std::ostream &s)
{
	vector<pair<float,float> > points(mesh->getPoints());
	int npoints = points.size();

    // set xrange [0:100]
    // set yrange [0:100]
    
	for(int i=0; i<npoints; i++)
    {
        s << points[i].first << " " << points[i].second << endl;
	}
    s << "e\n";

}

void GnuPlotVisualizer::drawFaces(std::ostream &s)
{
    vector<pair<int,int> > faces(mesh->getFaces());
    vector<pair<float,float> > points(mesh->getPoints());
    int nfaces = faces.size();
    
    // s << "plot '-' ls 2 with lines notitle\n";
    
    for(int i=0; i<nfaces; i++)
    {
        s << points[faces[i].first].first << " " 
          << points[faces[i].first].second;
        s << endl;
        s << points[faces[i].second].first << " " 
          << points[faces[i].second].second;
        s << endl;
    }
    
    s << "e\n";
}

void GnuPlotVisualizer::drawFaceNormals(std::ostream &s)
{
    vector<pair<int, int> > faces(mesh->getFaces());;
    vector<pair<float, float> > faceCentres(mesh->getFaceCentres());
    vector<pair<float, float> > faceNormals(mesh->getFaceNormals());
    int nfaces = faces.size();
    
    float endx, endy;
    
    for(int i=0; i<nfaces; i++)
    {
        endx = faceCentres[i].first + faceNormals[i].first * 0.1;
        endy = faceCentres[i].second + faceNormals[i].second * 0.1;
        
        s << "set arrow from " << faceCentres[i].first << "," 
          << faceCentres[i].second << " to " << endx << "," << endy 
          << " ls 1" << endl;
    }
}

void GnuPlotVisualizer::drawTrajectory(std::ostream &s, Particle particle)
{
    vector<pair<float, float> > track(particle.getTrajectory());
    int nsteps = track.size() - 1;
    
    for(int i=0; i<nsteps; i++)
    {
        s << "set arrow from " << track[i].first << "," << track[i].second
          << " to " << track[i+1].first << "," << track[i+1].second 
          << " ls 1" << endl;
    }
}

void GnuPlotVisualizer::drawParticleTracking(std::ostream &s, Particle & particle)
{
    vector<int>                 stepCells       (particle.getStepCells());
    vector<pair<int, float> >   cellFraction    (particle.getCellFraction());
    vector<pair<float, float> > track           (particle.getTrajectory());
    
    int nsteps = track.size() - 1;
    int nmoves = cellFraction.size();
    int movesPerStep = 0;
    
    // Vector ab is the vector from the beginning of the timestep to the end
    // of the timestep.
    float ax, ay;
    float bx, by;
    
    // Beginning of the line to draw. These points are used to draw the line 
    // with gnuplot.
    float lineBeginx, lineBeginy;
    
    // End of the line to draw.
    float lineEndx, lineEndy;
    
    // Fraction of the timestep on which tracking is done.
    float fractionDone;
    
    for (int i=0; i<nsteps; i++) 
    {
        ax = track[i].first;   ay = track[i].second;
        bx = track[i+1].first; by = track[i+1].second;
        fractionDone = 0;
        
        lineBeginx = ax;
        lineBeginy = ay;
        
        // Figure out the upper bound in the cellFraction array for use in the
        // inner loop below.
        if (i < (int)(stepCells.size()-1)) {
            movesPerStep = stepCells[i+1];
        } else {
            movesPerStep = nmoves;
        }

        for (int j=stepCells[i]; j<movesPerStep; j++)
        {
            s << "pause -1 'Hit OK to move to the next state'\n";
            s << "set title 'Particle in cell " << cellFraction[j].first 
              << "' font 'Arial,12'\n";
            s << "plot '-' ls 1 with lines notitle";
            s << ", '-' ls 3 with lines notitle\n";
            
            drawFaces(s);
            
            // lineEnd must point from (0 0) to the frac * ab (ab being the
            // vector pointing from a to b)
            // Therefore lineEnd = a + frac * (b - a)
            
            lineEndx = lineBeginx + cellFraction[j].second * (bx - ax);
            lineEndy = lineBeginy + cellFraction[j].second * (by - ay);
            
            // Write the results to the stream.
            s << lineBeginx << " " << lineBeginy;
            s << endl;
            s << lineEndx << " " << lineEndy << endl;
            s << "e\n";
            
            fractionDone += cellFraction[j].second;
            // std::cout << "Fraction done: " << fractionDone << endl;
            std::cout << "Fraction: " << cellFraction[j].second << std::endl;
            lineBeginx = lineEndx;
            lineBeginy = lineEndy;
        }
    }
    
}

void displayHelp(string programName)
{
    stringstream ss;
    ss << "Usage: " << programName << " [mesh directory] [particle directory] ";
    ss << "[particle label]" << endl;
    
    cout << ss.str();
}

int main(int argc, char** argv)
{
    // Argument processing
    
    string programName = argv[0];
    
    if (argc != 4) {
        displayHelp(programName);
        return 1;
    }
    
    string meshDirectory = argv[1];
    string particleDirectory = argv[2];
    
    stringstream helperSS(argv[3]);
    int particleLabel;
    
    if((helperSS >> particleLabel).fail())
    {
        cout << "Particle label must be an integer." << endl;
        displayHelp(programName);
        return 1;
    }
    
    // Read files
    
    Mesh *mesh = readMesh(meshDirectory);
    vector<Particle> particles(readParticles(particleDirectory, mesh));
    vector<pair<float,float> > track;
    
    
    GnuPlotVisualizer visualizer(mesh);
    std::ofstream gnuplotFile("output/mesh.cmd");
    // visualizer.drawPoints(gnuplotFile);
    visualizer.setStyle(gnuplotFile);
    visualizer.drawFaceNormals(gnuplotFile);
    visualizer.drawTrajectory(gnuplotFile, particles[particleLabel]);
    gnuplotFile <<  "plot '-' ls 2 with lines notitle\n";
    visualizer.drawFaces(gnuplotFile);
    
    particles[particleLabel].trackParticle();
    visualizer.drawParticleTracking(gnuplotFile, particles[particleLabel]);
    
    return 0;
}
