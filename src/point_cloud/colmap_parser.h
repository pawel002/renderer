#pragma once
#include <string>
#include <vector>

#include "objects.h"

std::vector<Point> readPoints3D(const std::string& file_path);
std::vector<CameraPose> readImages(const std::string& file_path);
