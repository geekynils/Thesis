#include <iostream>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <exception>

#include <cstdlib>

#include "particle/Particle.h"

// TODO rename to simply "Reader"

template <class T>
vector<pair<T,T> > readPairs(string path)
{
    ifstream file;
    
    file.exceptions (ifstream::eofbit | 
                     ifstream::failbit | 
                     ifstream::badbit );
    
    vector<pair<T, T> > v;
    T first, second;
    int n;
    
    try {
        file.open(path.c_str());
        file >> n;
        int i = 0;
        while (i < n) 
        {
            file >> first;
            file >> second;
            v.push_back(make_pair(first, second));
            i++;
        }
        file.close();
    }

    // TODO catch e per reference
    catch (ifstream::failure e) {
        cerr << "Exception when trying to read " << path << endl;
        exit(1);
    }
    
    return v;
}

template <class T>
vector<T> readValues(string path)
{
    ifstream file;
    
    file.exceptions (ifstream::eofbit | 
                     ifstream::failbit | 
                     ifstream::badbit );
    
    vector<T> v;
    T val;
    int n;
    
    try {
        file.open(path.c_str());
        file >> n;
        int i = 0;
        while (i < n) 
        {
            file >> val;
            v.push_back(val);
            i++;
        }
        file.close();
    }
    catch (ifstream::failure e) {
        cerr << "Exception when trying to read " << path << endl;
        exit(1);
    }
    
    return v;
}


vector<Particle> readParticles(string particleDir, Mesh *mesh);
Mesh* readMesh(string dir);
