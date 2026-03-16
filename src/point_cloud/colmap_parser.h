#pragma once
#include <string>
#include <vector>

#include "objects.h"

using namespace std;

vector<Point> readPoints3D(const string& file_path);
vector<CameraPose> readImages(const string& file_path);
