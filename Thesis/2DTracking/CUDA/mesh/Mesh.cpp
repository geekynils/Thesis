#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>

#include <mesh/Mesh.h>

using namespace std;

Mesh::Mesh(vector<pair<float, float> > points_,
           vector<pair<int, int> > faces_,
           vector<int> owners_,
           vector<int> neighbours_,
           vector<pair<float,float> > centroids_)
: points(points_), faces(faces_), owners(owners_), 
  neighbours(neighbours_), centroids(centroids_)
{
    calcFaceCentres();
    calcFaceNormals();
    flipFaceNormals();
}

Mesh::~Mesh()
{}

bool Mesh::isInternal(int faceLabel) 
{
    return neighbours[faceLabel] != -1 ? true : false;
}

bool Mesh::isOwner(int cellLabel, int faceLabel)
{
    if (owners[faceLabel] == cellLabel)
        return true;
    return false;
}

void Mesh::calcFaceCentres()
{
    float vecx, vecy;
    float point1x, point1y,
          point2x, point2y;

    int nfaces = faces.size();
    for (int i=0; i<nfaces; i++)
    {
        point1x = points[faces[i].first].first;
        point1y = points[faces[i].first].second;
        point2x = points[faces[i].second].first;
        point2y = points[faces[i].second].second;

        // Find a vector along the face
        vecx = point2x - point1x;
        vecy = point2y - point1y;

        // The face center is the addition of the
        // beginning point of the face and 1/2 of
        // the vector along the face calculated
        // above.
        faceCentres.push_back(
            make_pair(point1x + 0.5 * vecx,
                      point1y + 0.5 * vecy)
        );
    }
}

// TODO vectors point in normal direction but are not normalized
// not sure if this is required.
void Mesh::calcFaceNormals()
{
    float point1x, point1y,
          point2x, point2y;

    int nfaces = faces.size();
    for (int i=0; i<nfaces; i++)
    {
        point1x = points[faces[i].first].first;
        point1y = points[faces[i].first].second;
        point2x = points[faces[i].second].first;
        point2y = points[faces[i].second].second;
        
        faceNormals.push_back(
            calcFaceNormal(point1x, point1y, point2x, point2y)
        );
    }
}

/**
 * Flip the face normals so that they point out of the owner cell.
 */
void Mesh::flipFaceNormals()
{
    float vx, vy;
    float a;
    
    // For each face, find the centroid of the owner cell and
    // ensure that it points outside of the owner cell.
    
    int nfaces = faces.size();
    for (int i=0; i<nfaces; i++)
    {
        // (Cf - Centroid) * faceNormal < 0: normal points inside
        //                              > 0: normal points outside
        
        vx = faceCentres[i].first - centroids[owners[i]].first;
        vy = faceCentres[i].second  - centroids[owners[i]].second;
        a = vx * faceNormals[i].first + vy * faceNormals[i].second;
        
        if(a < 0)
        {
            // In this case we need to change the sign of the 
            // concerning face normal.
            faceNormals[i].first = - faceNormals[i].first;
            faceNormals[i].second = - faceNormals[i].second;
            
            // cout << "Flipping normal of face " << j << endl;
        }
    }
}

vector<int> Mesh::getCells()
{
    vector<int> cells;
    
    int nowners = owners.size();
    
    int currCell = 0;
    cells.push_back(currCell);
    
    for (int i=0; i<nowners; i++)
    {
        if (owners[i] != currCell)
        {
            currCell = owners[i];
            cells.push_back(currCell);
        }
    }
        
    return cells;
}

vector<int> Mesh::getSurroundingFaces(int cell)
{
    int nowners = owners.size();
    int nneighbours = neighbours.size();
    vector<int> surroundingFaces;
    
    for (int i=0; i<nowners; i++)
    {
        if(owners[i] == cell)
            surroundingFaces.push_back(i);
    }
    
    for (int i=0; i<nneighbours; i++)
    {
        if(neighbours[i] == cell)
            surroundingFaces.push_back(i);
    }
    
    return surroundingFaces;
}

bool Mesh::innerSide(int face, float x, float y)
{
    float Cfx = faceCentres[face].first;
    float Cfy = faceCentres[face].second;
    float Sfx = faceNormals[face].first;
    float Sfy = faceNormals[face].second;
    
    // Vector from the face centre to the given point.
    float vx, vy;
    vx = x - Cfx;
    vy = y - Cfy;
    
    // v * Sf
    // ------   = cos(theta)
    //|v| |Sf|
    
    // theta is the angle between v and Sf

    // We denote cos(theta) with a.
    float a = (vx * Sfx + vy * Sfy) 
            / ( sqrt(vx*vx + vy*vy) * sqrt(Sfx*Sfx + Sfy*Sfy) );
    
    // Let's include the theoretical case in which the particle is exactly 
    // on the face.
    if(a <= 0) {
        return true;
    }
    
    return false;
}

/**
 * Returns the label of the cell in which the point resides.
 * -1 if it's outside of the mesh.
 */
// TODO Points which are close to a face get usually -1
int Mesh::findCell(float x, float y)
{
    vector<int> cells(getCells());

    int nowners = owners.size();
    int nneighbours = neighbours.size();
    int ncells = cells.size();
    
    // Does the point lie on the inner side of all owned faces?
    bool ownersOk = true;
    
    // Does the point NOT lie on the inner side of all neighbour faces?
    bool neighbourOk = true;
    
    for(int i=0; i<ncells; i++)
    {
        ownersOk = true;
        neighbourOk = true;
        
        for (int j=0; j<nowners; j++)
        {
            if(owners[j] == cells[i])
            {
                // The point must be on the inner side of all owned faces.
                if (!innerSide(j, x, y)) {
                    ownersOk = false;
                    break;
                }
            }
        }
        
        if(ownersOk) {
            for(int j=0; j<nneighbours; j++)
            {
                if (neighbours[j] == cells[i])
                {
                    // If the point is on the inner side of one of the neighbour
                    // faces then it's not in this cell.
                    if (innerSide(j, x, y)) {
                        neighbourOk = false;
                        break;
                    }
                }
            }
        }
        
        if (neighbourOk && ownersOk) {
            return i;
        }
    }
    
    return -1;
}


// Functions to access the vectors as array
// Syntax: Reference to a pointer
int Mesh::getPointsx(float *&pointsx)
{
    int npoints = points.size();
    pointsx = new float[npoints];
    for (int i=0; i<npoints; i++) {
        pointsx[i] = points[i].first;
    }
    return npoints;
}

int Mesh::getPointsy(float *&pointsy)
{
    int npoints = points.size();
    pointsy = new float[npoints];
    for (int i=0; i<npoints; i++) {
        pointsy[i] = points[i].second;
    }
    return npoints;
}

int Mesh::getFacesStart(int *&facesStart)
{
    int nfaces = faces.size();
    facesStart = new int[nfaces];
    for (int i=0; i<nfaces; i++) {
        facesStart[i] = faces[i].first;
    }
    return nfaces;
}

int Mesh::getFacesEnd(int *&facesEnd)
{
    int nfaces = faces.size();
    facesEnd = new int[nfaces];
    for (int i=0; i<nfaces; i++) {
        facesEnd[i] = faces[i].second;
    }
    return nfaces;
}

int Mesh::getFaceCentresx(float *&faceCentresx)
{
    int nfaces = faces.size();
    faceCentresx = new float[nfaces];
    for (int i=0; i<nfaces; i++) {
        faceCentresx[i] = faceCentres[i].first;
    }
    return nfaces;
}

int Mesh::getFaceCentresy(float *&faceCentresy)
{
    int nfaces = faces.size();
    faceCentresy = new float[nfaces];
    for (int i=0; i<nfaces; i++) {
        faceCentresy[i] = faceCentres[i].second;
    }
    return nfaces;
}

int Mesh::getFaceNormalsx(float *&faceNormalsx)
{
    int nfaces = faces.size();
    faceNormalsx = new float[nfaces];
    for (int i=0; i<nfaces; i++) {
        faceNormalsx[i] = faceNormals[i].first;
    }
    return nfaces;
}

int Mesh::getFaceNormalsy(float *&faceNormalsy)
{
    int nfaces = faces.size();
    faceNormalsy = new float[nfaces];
    for (int i=0; i<nfaces; i++) {
        faceNormalsy[i] = faceNormals[i].second;
    }
    return nfaces;
}

int Mesh::getOwners(int *&owners_a)
{
    int nowners = owners.size();
    owners_a = new int[nowners];
    for (int i=0; i<nowners; i++) {
        owners_a[i] = owners[i];
    }
    return nowners;
}

int Mesh::getNeighbours(int *&neighbours_a)
{
    int nneighbours = neighbours.size();
    neighbours_a = new int[nneighbours];
    for (int i=0; i<nneighbours; i++) {
        neighbours_a[i] = neighbours[i];
    }
    return nneighbours;
}

int Mesh::getCentroidsx(float *&centroids_x)
{
    int ncells = centroids.size();
    centroids_x = new float[ncells];
    for (int i=0; i<ncells; i++) {
        centroids_x[i] = centroids[i].first;
    }
    return ncells;
}

int Mesh::getCentroidsy(float *&centroids_y)
{
    int ncells = centroids.size();
    centroids_y = new float[ncells];
    for (int i=0; i<ncells; i++) {
        centroids_y[i] = centroids[i].second;
    }
    return ncells;
}


int Mesh::getFacesIndexedByCells(
		int *&owned_faces_index,
		int *&owned_faces_a,
		int *&neighbour_faces_index,
		int *&neighbour_faces_a,

		// Size of the arrays
		// n_owned_faces_index and n_neighbour_faces_index equals ncells
		int &n_owned_faces,
		int &n_neighbour_faces
)
{
	// List containing all cell labels
	vector<int> cells;
	bool included;
	for(int i=0; i<owners.size(); i++)
	{
		included = false;
		for(int j=0; j<cells.size(); j++)
		{
			// Check if we already added the cell to the cells vector.
			if(cells[j] == owners[i]) {
				included = true;
				break;
			}
		}
		if(!included)
		{
			cells.push_back(owners[i]);
		}
	}

	sort(cells.begin(), cells.end());
	int ncells = cells.size();

	// Holds the cell label and the corresponding owned face.
	// Cell labels are repeatedly in the vector.
	vector<pair<int, int> > owned_faces;
	for(int i=0; i<ncells; i++)
	{
		for(int j=0; j<owners.size(); j++)
		{
			if(cells[i] == owners[j]) {
				owned_faces.push_back(make_pair(cells[i], j));
			}
		}
	}

	// Does the cell have any neighbour cells?
	// If not we write -1 in the index.
	bool hasNeighbour;
	vector<pair<int, int> > neighbour_faces;
	for(int i=0; i<ncells; i++)
	{
		hasNeighbour = false;
		for(int j=0; j<neighbours.size(); j++)
		{
			if(cells[i] == neighbours[j])
			{
				hasNeighbour = true;
				neighbour_faces.push_back(make_pair(cells[i], j));
			}
		}

		if(!hasNeighbour) {
			neighbour_faces.push_back(make_pair(-1,0));
		}
	}



	n_owned_faces = owned_faces.size();
	n_neighbour_faces =  0;

	for(int i=0; i<neighbour_faces.size(); i++)
	{
		if(neighbour_faces[i].first != -1)
			n_neighbour_faces++;
	}

	// Allocate memory for the arrays.
	// Note index is the small array to access the large one correctly.

	owned_faces_index = new int[ncells];
	owned_faces_a = new int[n_owned_faces];
	neighbour_faces_index = new int[ncells];
	neighbour_faces_a = new int[n_neighbour_faces];

	int lastCell = -1;
	// Index of the index array
	int j = 0;
	for(int i=0; i<n_owned_faces; i++)
	{
		// Update index?
		if(owned_faces[i].first != lastCell)
		{
			owned_faces_index[j] = i;
			lastCell = owned_faces[i].first;
			j++;
		}
		owned_faces_a[i] = owned_faces[i].second;
	}

	lastCell = -2;
	j = 0;
	int neighbour_faces_a_idx = 0;
	for(int i=0; i<neighbour_faces.size(); i++)
	{
		// Case where the cell does not have any neighbour faces.
		if(neighbour_faces[i].first == -1)
		{
			neighbour_faces_index[i] = -1;
			continue;
		}
		if(neighbour_faces[i].first != lastCell)
		{
			neighbour_faces_index[i] = j;
			lastCell = j;
			j++;
		}
		neighbour_faces_a[neighbour_faces_a_idx] = neighbour_faces[i].second;
		neighbour_faces_a_idx++;
	}

	return ncells;
}

int Mesh::getMeshFlat(float **mesh_flat, bool swapNormals)
{
    vector<int> cells = getCells();
    int ncells = cells.size();
    int nfaces = faceCentres.size();
    
    int mesh_flat_len = ncells * 32;
    
    // TODO use new
    *mesh_flat = static_cast<float*>(malloc(sizeof(float) * mesh_flat_len));
    
    // First we initialize everything with -1
    for (int i=0; i<mesh_flat_len; i++)
        (*mesh_flat)[i] = -1;
    
    // Fill in the cell labels
    // Cell labels are at 0, 32, 64 etc..
    int k=0;
    for(int i=0; i<32*ncells; i+=32)
    {
        (*mesh_flat)[i] = cells[k];
        k++;
    }
    
    // Temp vector containing the face indices of the faces belonging to the
    // current cell.
    vector<int> facesBelongingToCell;
    vector<int> toSwap;
    
    for(int i=0; i<ncells; i++)
    {
        facesBelongingToCell.clear();
        toSwap.clear();
        
        // Fill in the owner face labels.
        k=0;
        for(int j=0; j<nfaces; j++)
        {
            // Which face labels belong to this cell
            if (owners[j] == i)
            {
                // Cell starts at 32*i
                // Position of the face label: 5*k
                (*mesh_flat)[32*i + 1 + 5*k] = j;
                k++;
                facesBelongingToCell.push_back(j);
            }
        }
        
        // Fill in the neighbour face labels.
        for(unsigned int j=0; j<neighbours.size(); j++)
        {
            if (neighbours[j] == i)
            {
                (*mesh_flat)[32*i + 1 + 5*k] = j;
                k++;
                facesBelongingToCell.push_back(j);
                toSwap.push_back(j);
            }
        }
        
        // Fill in the face centres and normals.
        // TODO swap normals
        for(unsigned int j=0; j<facesBelongingToCell.size(); j++)
        {
            // 32*i                 + 2                                                + 5*j
            // ^ Start of the cell    ^ First field used for first cell and face label   ^ Start of the face centre entry
            (*mesh_flat)[32*i + 2 + 5*j]     = faceCentres[facesBelongingToCell[j]].first;
            (*mesh_flat)[32*i + 2 + 5*j + 1] = faceCentres[facesBelongingToCell[j]].second;
            
            if(swapNormals)
            {
                for(unsigned int l=0; l<toSwap.size(); l++)
                {
                    if(facesBelongingToCell[j] == toSwap[l])
                    {
                        (*mesh_flat)[32*i + 2 + 5*j + 2] = (-1) * faceNormals[facesBelongingToCell[j]].first;
                        (*mesh_flat)[32*i + 2 + 5*j + 3] = (-1) * faceNormals[facesBelongingToCell[j]].second;
                    }
                }
            } else {
                (*mesh_flat)[32*i + 2 + 5*j + 2] = faceNormals[facesBelongingToCell[j]].first;
                (*mesh_flat)[32*i + 2 + 5*j + 3] = faceNormals[facesBelongingToCell[j]].second;
            }

        }
    }
    
    return mesh_flat_len;
}

