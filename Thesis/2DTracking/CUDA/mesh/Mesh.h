#include <vector>
#include <utility>

#include <gtest/gtest_prod.h>

using namespace std;

#ifndef MESH_H_
#define MESH_H_


/**
 * 2D unstructured mesh class
 */
class Mesh
{

private:
	vector<pair<float,float> > points;

	// Contains the two points at which a face begins and ends
	vector<pair<int, int> > faces;

    // Owners: Index matches face label, values are cell labels
    vector<int> owners;
    
    // Neighbours: -1 If the cell lies on the boundary and has
    // no neighbour
    vector<int> neighbours;
    
    // Instead of calculating the centroids, we read it for now.
    // The centroids are not really centroids, but just a point lying in the 
    // cell.
    vector<pair<float, float> > centroids;

	// Face centres and normals for the concerning faces
	vector<pair<float, float> > faceCentres;
	vector<pair<float, float> > faceNormals;
    
    void calcFaceCentres();
    void calcFaceNormals();
    void flipFaceNormals();
    
    /**
     * Figure out if a point lies on the "inner" side of a face.
     * The inner side is the side on which the owner cell lies.
     */
    bool innerSide(int face, float x, float y);
    
    inline pair<float, float> calcFaceNormal
        (float point1x, float point1y, 
         float point2x, float point2y)
    {
        float vecx, vecy;
        
        // Find the normal accross a straight line.
        // Take the line as vector [x y]'
        // x = xend - xbegin
        // y = yend - ybegin
        // Then rotate it around 90 degree in anticlockwise
        // direction.
        // x' = x cos(t) - y sin(t)
        // y' = x sin(t) + y cos(t)
        // Recall cos(t=90) = 0 and sin(t=90) = 1
        // Therefore x' = -y and y' = x
        
        // Find a vector along the face
        vecx = point2x - point1x;
        vecy = point2y - point1y;
        
        return make_pair(-vecy, vecx);
    }

public:

    Mesh(vector<pair<float, float> > points_,
         vector<pair<int, int> > faces_,
         vector<int> owners_,
         vector<int> neighbours_,
         vector<pair<float,float> > centroids_);

	virtual ~Mesh();
    
    /**
     * Returns true if the concerning face is an internal face.
     */
    bool isInternal(int faceLabel);
    
    /**
     * Returns true if the cell is owner of the face
     */
    bool isOwner(int cellLabel, int faceLabel);
    
    /**
     * Search over all cells, reports in which cell the given point is.
     */
    int findCell(float x, float y);
    
    /**
     * Returns a vector containing all cell labels.
     */
    vector<int> getCells();
    
    /**
     * Returns the labels of all faces surrounding a cell.
     */
    vector<int> getSurroundingFaces(int cell);

    /**
     * Returns arrays which enable direct access to all the faces surrounding a
     * cell. The parameters are allocated in the method, use null pointers when
     * calling the method.
     *
     * @param owned_faces_index The array indices correspond to cell labels.
     * 		  Holds the beginning indices of the owned_faces array.
     *
     * @param owned_faces Values Correspond to the labels of the owned faces.
     *
     * @param neighbour_faces_index Same Idea.
     *
     * @return number of cells
     */
    int getFacesIndexedByCells(
    		int *&owned_faces_index,
    		int *&owned_faces_a,
    		int *&neighbour_faces_index,
    		int *&neighbour_faces_a,

    		// Size of the arrays
    		// n_owned_faces_index and n_neighbour_faces_index equals ncells
    		int &n_owned_faces,
    		int &n_neighbour_faces
    );

    /**
     * @param pointsx pointer to float array
     * @return number of points
     */
    int getPointsx(float *&pointsx);
    
    int getPointsy(float *&pointsy);
    
    int getFacesStart(int *&facesStart);
    
    int getFacesEnd(int *&facesEnd);
    
    int getFaceCentresx(float *&faceCentresx);
    
    int getFaceCentresy(float *&faceCentresy);
    
    int getFaceNormalsx(float *&faceNormalsx);
    
    int getFaceNormalsy(float *&faceNormalsy);
    
    int getOwners(int *&owners);
    
    int getNeighbours(int *&neighbours_a);
    
    int getCentroidsx(float *&centroids_x);
    
    int getCentroidsy(float *&centroids_y);
    
    /**
     * Returns a flat array containing the whole mesh. 
     * Each cell gets 32 elements (= 128 bytes)
     * So cell 0 starts at 0, cell 1 at 32, etc
     * Order of elements is: Cell ID, face ID, face centre, face normal, face id, ..
     * For every cell, we assume 4 faces. If a cell has less faces values are 
     * set to -1 after the last face.
     * mesh_flat will be allocated, don't forget to free afterwards.
     *
     * @param swapNormals If set to true, normals of neighbour faces are swapped
     *                    so they always point out of the cell.
     * 
     * @return length of the array
     */
    int getMeshFlat(float **mesh_flat, bool swapNormals);
    
	inline vector<pair<float,float> > getPoints()	{
		return points;
	}

	inline vector<pair<int,int> > getFaces() {
		return faces;
	}
    
    inline vector<pair<float, float> > getFaceCentres() {
        return faceCentres;
    }
    
    inline vector<pair<float, float> > getFaceNormals() {
        return faceNormals;
    }
    
    inline vector<pair<float, float> > getCentroids() {
        return centroids;
    }
    
    inline vector<int> getOwners() {
        return owners;
    }
    
    inline vector<int> getNeighbours() {
        return neighbours;
    }
    
    FRIEND_TEST(MeshTest, testCalcFaceNormals);
    FRIEND_TEST(MeshTest, innerSide);
};


#endif /* MESH_H_ */
