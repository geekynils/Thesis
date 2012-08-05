#include <cmath>
#include <algorithm>

#include "mesh/Mesh.h"
#include "mesh/MeshReader.h"

#include <gtest/gtest.h>

TEST(MeshTest, testCalcFaceNormals)
{
	Mesh *mesh = readMesh("mesh");
    
    float x1, x2, y1, y2;
    // Expected Normal
    float enx, eny;
    // Calculated Normal
    float cnx, cny;
    // Length
    float len;
    pair<float, float> cn;
    
    
    // --- First line
    x1 = 0; y1 = 0;
    x2 = 1; y2 = 0;
    enx = 0; eny = 1;
    
    cn = mesh->calcFaceNormal(x1, y1, x2, y2);
    cnx = cn.first; cny = cn.second;
    
    // Normalize length
    len = sqrt((cnx*cnx + cny*cny));
    cnx /= len; cny /= len;
    // Direction should not matter
    cnx = abs(cnx); cny = abs(cny);
    
    ASSERT_FLOAT_EQ(enx, cnx);
    ASSERT_FLOAT_EQ(eny, cny);
    
    // --- Second line
    x1 = 3; y1 = 3;
    x2 = 16; y2 = 8;
    enx = 0.358979; eny = 0.93334556; // must be normalized
    
    cn = mesh->calcFaceNormal(x1, y1, x2, y2);
    cnx = cn.first; cny = cn.second;
    
    // Normalize length
    len = sqrt((cnx*cnx + cny*cny));
    cnx /= len; cny /= len;
    // Direction should not matter
    cnx = abs(cnx); cny = abs(cny);
    
    ASSERT_FLOAT_EQ(enx, cnx);
    ASSERT_FLOAT_EQ(eny, cny);
    
    // --- Second line point 1 and 2 interchanged
    x1 = 16; y1 = 8;
    x2 = 3; y2 = 3;
    enx = 0.358979; eny = 0.93334556; // must be normalized
    
    cn = mesh->calcFaceNormal(x1, y1, x2, y2);
    cnx = cn.first; cny = cn.second;
    
    // Normalize length
    len = sqrt((cnx*cnx + cny*cny));
    cnx /= len; cny /= len;
    // Direction should not matter
    cnx = abs(cnx); cny = abs(cny);
    
    ASSERT_FLOAT_EQ(enx, cnx);
    ASSERT_FLOAT_EQ(eny, cny);
}

TEST(MeshTest, internalFace)
{
    Mesh *mesh = readMesh("mesh");
    ASSERT_TRUE(mesh->isInternal(1));
    ASSERT_TRUE(mesh->isInternal(1));
    ASSERT_FALSE(mesh->isInternal(7));
    ASSERT_FALSE(mesh->isInternal(0));
}

TEST(MeshTest, getCells)
{
    Mesh *mesh = readMesh("mesh");
    vector<int> cells = mesh->getCells();
    int ncells = cells.size();
    ASSERT_EQ(3, ncells);
}

TEST(MeshTest, getSurroundingFaces)
{
    Mesh *mesh = readMesh("mesh");
    vector<int> surroundingFaces(mesh->getSurroundingFaces(1));
    int nfaces = surroundingFaces.size();
    sort(surroundingFaces.begin(), surroundingFaces.end());
    
    ASSERT_EQ(3, nfaces);
    ASSERT_EQ(1, surroundingFaces[0]);
    ASSERT_EQ(4, surroundingFaces[1]);
    ASSERT_EQ(5, surroundingFaces[2]);
}

TEST(MeshTest, innerSide)
{
    Mesh *mesh = readMesh("mesh");
    
    ASSERT_TRUE(mesh->innerSide(1, 2.5, 2.5));
    ASSERT_TRUE(mesh->innerSide(5, 4, 1));
    ASSERT_FALSE(mesh->innerSide(2, 2.5, 3.5));
}

TEST(MeshTest, findCell)
{
    Mesh *mesh = readMesh("mesh");
    ASSERT_EQ(0, mesh->findCell(2,2));
    ASSERT_EQ(-1, mesh->findCell(0,0));
    ASSERT_EQ(1, mesh->findCell(4,1));
    ASSERT_EQ(2, mesh->findCell(3,3.5));
}

TEST(MeshTest, findCell2)
{
    Mesh *mesh = readMesh("mesh");
    ASSERT_EQ(2, mesh->findCell(2.5,3.5));
}

TEST(MeshTest, getFacesIndexedByCells)
{
	int expected_owned_faces_index[]    = {0, 4, 6};
	int expected_owned_faces_a[]        = { 0, 1, 2, 3, 4, 5, 6, 7, 8 };
	int expected_neighbour_faces_index[]= {-1, 0, 1 };
	int expected_neighbour_faces_a[]    = {1, 2};

	int expected_n_owned_faces			= 9;
	int expected_n_neighbour_faces		= 2;
	int expected_ncells					= 3;

	int *owned_faces_index;
	int *owned_faces_a;
	int *neighbour_faces_index;
	int *neighbour_faces_a;

	int n_owned_faces;
	int n_neighbour_faces;

	Mesh *mesh = readMesh("mesh");

	int ncells = mesh->getFacesIndexedByCells
	(
		owned_faces_index,
		owned_faces_a,
		neighbour_faces_index,
		neighbour_faces_a,
		n_owned_faces,
		n_neighbour_faces
	);

	ASSERT_EQ(expected_ncells, ncells);
	ASSERT_EQ(expected_n_owned_faces, n_owned_faces);
	ASSERT_EQ(expected_n_neighbour_faces, n_neighbour_faces);

	for(int i=0; i<ncells; i++)
	{
		ASSERT_EQ(expected_owned_faces_index[i], owned_faces_index[i]);
	}

	for(int i=0; i<ncells; i++)
	{
		ASSERT_EQ(expected_neighbour_faces_index[i], neighbour_faces_index[i]);
	}

	for(int i=0; i<n_owned_faces; i++) {
		ASSERT_EQ(expected_owned_faces_a[i], owned_faces_a[i]);
	}

	for(int i=0; i<n_neighbour_faces; i++) {
		ASSERT_EQ(expected_neighbour_faces_a[i], neighbour_faces_a[i]);
	}
}
