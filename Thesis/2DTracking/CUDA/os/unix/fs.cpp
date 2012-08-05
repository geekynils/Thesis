#include <vector>
#include <string>
#include <iostream>

#include <cstddef>
#include <sys/types.h>
#include <dirent.h>

#include "fs.h"

using namespace std;

vector<string> listFiles(string path)
{
    DIR *dp;
    struct dirent *ep;
    vector<string> files;

    dp = opendir (path.c_str());
    if (dp != NULL)
    {
        while ((ep = readdir(dp)))
        {
        	string s(ep->d_name);
        	if(!(s[0] == '.'))		// Omit invisible files and .. or .
        		files.push_back(s);
        }
        closedir (dp);
    } else {
        cerr << "Couldn't open the directory" << endl;
    }

    return files;
}
