#pragma once
#include <string>
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

struct Vertex {
    glm::vec3 position;
    glm::vec3 color;
};

struct CameraPose {
    glm::mat4 modelMatrix;
};

std::vector<Vertex> readPoints3D(const std::string& filepath);
std::vector<CameraPose> readImages(const std::string& filepath);